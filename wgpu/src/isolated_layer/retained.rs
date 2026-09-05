//! Renderer-local retention of final, pre-composite isolated-layer outputs.
//!
//! A [`SurfaceHandle`](isolated_layer::SurfaceHandle) names exactly one output
//! slot. The slot never contains source and effect-result variants: replacing
//! its key atomically replaces its sole committed target.

use super::{Context, Target, effect};
use crate::core::{Rectangle, isolated_layer};

use std::collections::{BTreeMap, HashMap};

/// An exact signature for the pixels in one retained output slot.
///
/// Fixed-function composition is deliberately absent. It happens after the
/// cached output and therefore cannot change the retained pixels.
#[derive(Debug, Clone)]
pub(crate) struct OutputKey {
    content: isolated_layer::ContentStamp,
    raster: RasterDescriptor,
    absolute_geometry: Option<AbsoluteGeometry>,
    evidence: effect::LayerInputEvidence,
}

impl OutputKey {
    /// Captures every renderer-owned input needed to validate a final output.
    ///
    /// `position_sensitive` must be true when either the captured child or any
    /// effect can make its pixels depend on absolute layer position.
    pub(crate) fn new(
        request: &isolated_layer::CacheRequest,
        evidence: effect::LayerInputEvidence,
        context: &Context,
        content_bounds: Rectangle,
        position_sensitive: bool,
        device_epoch: u64,
    ) -> Self {
        Self {
            content: request.stamp().clone(),
            raster: RasterDescriptor::new(context, device_epoch),
            absolute_geometry: position_sensitive.then(|| AbsoluteGeometry {
                represented_bounds: rectangle_bits(context.represented_bounds),
                content_bounds: rectangle_bits(content_bounds),
            }),
            evidence,
        }
    }

    /// Compares complete keys, including retained effect-input snapshots.
    ///
    /// Volatile evidence never matches, including an otherwise equal snapshot.
    pub(crate) fn matches(&self, other: &Self) -> bool {
        self.first_mismatch(other).is_none()
    }

    fn first_mismatch(&self, other: &Self) -> Option<KeyMismatch> {
        if self.content != other.content {
            Some(KeyMismatch::Content)
        } else if self.raster != other.raster {
            Some(KeyMismatch::Raster)
        } else if self.absolute_geometry != other.absolute_geometry {
            Some(KeyMismatch::Geometry)
        } else if !self.evidence.matches(&other.evidence) {
            Some(KeyMismatch::EffectInputs)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RasterDescriptor {
    source_rect: [u32; 4],
    logical_surface_size_bits: [u32; 2],
    source_content_relative_bits: [u32; 4],
    represented_size_bits: [u32; 2],
    physical_viewport: [u32; 2],
    backing_extent: [u32; 2],
    scale_factor_bits: u32,
    format: wgpu::TextureFormat,
    device_epoch: u64,
}

impl RasterDescriptor {
    fn new(context: &Context, device_epoch: u64) -> Self {
        let physical_viewport = context.physical_viewport();
        let backing_extent = context.backing_extent();

        Self {
            source_rect: [
                context.source_rect.x,
                context.source_rect.y,
                context.source_rect.width,
                context.source_rect.height,
            ],
            logical_surface_size_bits: context.logical_surface_size_bits,
            source_content_relative_bits: context.source_content_relative_bits,
            represented_size_bits: [
                canonical_bits(context.represented_bounds.width),
                canonical_bits(context.represented_bounds.height),
            ],
            physical_viewport: [physical_viewport.width, physical_viewport.height],
            backing_extent: [backing_extent.width, backing_extent.height],
            scale_factor_bits: canonical_bits(context.scale_factor()),
            format: context.format,
            device_epoch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AbsoluteGeometry {
    represented_bounds: [u32; 4],
    content_bounds: [u32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyMismatch {
    Content,
    Raster,
    Geometry,
    EffectInputs,
}

/// Opaque ownership token for one in-flight output-slot access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LeaseTicket(u64);

/// Why a retained output was not reusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputMiss {
    /// No registry metadata exists for the output identity.
    UnknownIdentity,
    /// Metadata exists, but its target is not resident.
    Targetless,
    /// The request supplied no caller-managed content evidence.
    MissingContentEvidence,
    /// Duplicate content observations disagreed on generation.
    ContentConflict,
    /// The request changed between construction and lookup.
    ContentChangedBeforeLookup,
    /// The observation is older than a generation already seen by this slot.
    StaleContent,
    /// The observation both advances and regresses previously seen inputs.
    IncomparableContent,
    /// The observed content signature changed.
    ContentChanged,
    /// A target rasterization fact changed.
    RasterChanged,
    /// Absolute geometry changed for a position-sensitive producer.
    GeometryChanged,
    /// Exact effect or stack input evidence changed.
    EffectInputsChanged,
    /// At least one effect declined retained-output reuse.
    VolatileInputs,
    /// More than one producer tried to write the same slot concurrently.
    CompetingWriter,
}

/// Result of leasing a retained output slot.
pub(crate) struct OutputLease<T = Target> {
    /// A valid committed target on a hit; absent on a miss.
    pub(crate) target: Option<T>,
    /// The token required to return a hit or commit a rendered miss.
    pub(crate) ticket: Option<LeaseTicket>,
    /// Whether `target` contains pixels matching the requested key.
    pub(crate) valid: bool,
    /// Whether the caller may attempt to store its target using `ticket`.
    pub(crate) cacheable: bool,
    /// The miss reason, absent on a hit.
    pub(crate) miss: Option<OutputMiss>,
    /// Whether same-frame residency requests disagreed on priority.
    pub(crate) priority_conflict: bool,
}

/// Result of applying an identity-only keep-alive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeepAliveOutcome {
    /// Whether registry metadata existed for the identity.
    pub(crate) found: bool,
    /// Whether same-frame observations disagreed on priority.
    pub(crate) priority_conflict: bool,
}

/// Why a rendered target was not committed to its output slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreDisposition {
    /// The target and its complete key were committed together.
    Stored,
    /// The output metadata was swept before the store.
    UnknownEntry,
    /// The ticket does not own the slot's active lease.
    TicketMismatch,
    /// The lease belongs to a different rendered frame.
    FrameMismatch,
    /// Another producer contended for the slot while this lease was active.
    CompetingWriter,
    /// The cache request does not describe the leased output slot and stamp.
    RequestMismatch,
    /// Caller-managed content changed while the lease was active.
    ContentChanged,
    /// Rasterization facts changed while the lease was active.
    RasterChanged,
    /// Position-sensitive absolute geometry changed while the lease was active.
    GeometryChanged,
    /// Effect or stack inputs changed while the lease was active.
    EffectInputsChanged,
    /// Recollected effect evidence is volatile.
    VolatileInputs,
}

/// Result of returning or committing a leased target.
pub(crate) struct StoreOutcome<T = Target> {
    /// Whether the target was stored and, otherwise, why it was rejected.
    pub(crate) disposition: StoreDisposition,
    /// Targets no longer owned by the registry.
    pub(crate) released: Vec<T>,
}

impl<T> StoreOutcome<T> {
    /// Returns whether the supplied target was committed.
    pub(crate) fn stored(&self) -> bool {
        self.disposition == StoreDisposition::Stored
    }
}

/// Abandoned leases cleared at a rendered-frame boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RecoveryOutcome {
    /// Number of output leases cleared.
    pub(crate) output_leases: usize,
}

/// Active leases found at the end of one rendered frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LeaseAudit {
    /// Number of current-frame output leases still active.
    pub(crate) output_leases: usize,
}

/// Entries removed by a liveness sweep.
pub(crate) struct SweepOutcome<T = Target> {
    /// Resident targets removed with their metadata.
    pub(crate) released: Vec<T>,
    /// Number of output-slot metadata entries removed.
    pub(crate) outputs: usize,
}

/// Memory-pressure eviction tier of a retained output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EvictionTier {
    /// Ordinary cached output.
    Normal,
    /// Output requested with protected residency priority.
    Protected,
}

/// One target evicted under the renderer-owned byte budget.
pub(crate) struct EvictedTarget<T = Target> {
    /// Evicted target ownership.
    pub(crate) target: T,
    /// Residency tier used for eviction ordering and diagnostics.
    pub(crate) tier: EvictionTier,
}

/// Result of enforcing a retained-target byte limit.
pub(crate) struct EvictionOutcome<T = Target> {
    /// Targets evicted in policy order.
    pub(crate) evicted: Vec<EvictedTarget<T>>,
    /// Resident bytes after eviction.
    pub(crate) remaining_bytes: u64,
}

/// Resident bytes split by priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ResidentBytes {
    /// Normal-priority bytes.
    pub(crate) normal: u64,
    /// Protected-priority bytes.
    pub(crate) protected: u64,
}

impl ResidentBytes {
    /// Returns all bytes owned by committed retained targets.
    pub(crate) fn total(self) -> u64 {
        self.normal.saturating_add(self.protected)
    }
}

/// Bounded-cardinality registry metadata counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct MetadataCounts {
    /// Output-slot metadata entries.
    pub(crate) outputs: usize,
    /// Slots whose sole committed output is resident.
    pub(crate) resident_outputs: usize,
    /// Per-content-identity monotonic watermark records.
    pub(crate) watermark_records: usize,
}

/// The byte-size operation needed by the generic registry implementation.
///
/// The generic parameter exists only so ownership behavior can be tested
/// without constructing GPU resources. Production code uses [`Target`].
pub(crate) trait OutputTarget {
    /// Returns the target's renderer-owned allocation size.
    fn output_byte_size(&self) -> u64;
}

impl OutputTarget for Target {
    fn output_byte_size(&self) -> u64 {
        self.byte_size()
    }
}

struct CommittedOutput<T> {
    key: OutputKey,
    target: T,
}

#[derive(Debug, Clone)]
struct ActiveLease {
    ticket: LeaseTicket,
    frame: u64,
    expected: OutputKey,
    kind: LeaseKind,
    contended: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeaseKind {
    Hit,
    Replacement,
}

struct Entry<T> {
    committed: Option<CommittedOutput<T>>,
    active: Option<ActiveLease>,
    latest_stamp: isolated_layer::ContentStamp,
    watermarks: BTreeMap<u64, u64>,
    priority: isolated_layer::CacheResidencyPriority,
    priority_frame: u64,
    priority_conflict_frame: Option<u64>,
    marked_frame: u64,
    last_used: u64,
}

impl<T> Entry<T> {
    fn new(
        request: &isolated_layer::CacheRequest,
        key: OutputKey,
        ticket: LeaseTicket,
        frame: u64,
    ) -> Self {
        let mut entry = Self {
            committed: None,
            active: Some(ActiveLease {
                ticket,
                frame,
                expected: key,
                kind: LeaseKind::Replacement,
                contended: false,
            }),
            latest_stamp: request.stamp().clone(),
            watermarks: BTreeMap::new(),
            priority: request.priority(),
            priority_frame: frame,
            priority_conflict_frame: None,
            marked_frame: frame,
            last_used: frame,
        };
        entry.observe_watermarks(request.stamp());
        entry
    }

    fn observe_watermarks(&mut self, stamp: &isolated_layer::ContentStamp) {
        for revision in stamp.revisions() {
            let _ = self
                .watermarks
                .entry(revision.identity())
                .and_modify(|generation| {
                    *generation = (*generation).max(revision.generation());
                })
                .or_insert(revision.generation());
        }
    }

    fn accept_stamp(&mut self, stamp: &isolated_layer::ContentStamp) {
        self.observe_watermarks(stamp);
        self.latest_stamp = stamp.clone();
    }

    fn observe_priority(
        &mut self,
        priority: isolated_layer::CacheResidencyPriority,
        frame: u64,
    ) -> bool {
        self.marked_frame = frame;

        if self.priority_frame != frame {
            self.priority = priority;
            self.priority_frame = frame;
            self.priority_conflict_frame = None;
            return false;
        }

        if self.priority != priority || self.priority_conflict_frame == Some(frame) {
            self.priority = isolated_layer::CacheResidencyPriority::Normal;
            self.priority_conflict_frame = Some(frame);
            true
        } else {
            false
        }
    }
}

/// A renderer-local registry containing at most one committed output per
/// surface identity.
pub(crate) struct Registry<T: OutputTarget = Target> {
    entries: HashMap<u64, Entry<T>>,
    next_ticket: u64,
}

impl<T: OutputTarget> Default for Registry<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            next_ticket: 0,
        }
    }
}

impl<T: OutputTarget> Registry<T> {
    /// Marks an existing output identity alive without changing its key.
    ///
    /// Keep-alives never create registry metadata and never provide validity
    /// evidence. Same-frame priority disagreement deterministically resolves
    /// to normal priority.
    pub(crate) fn keep_alive(
        &mut self,
        keep_alive: &isolated_layer::CacheKeepAlive,
        frame: u64,
    ) -> KeepAliveOutcome {
        let Some(entry) = self.entries.get_mut(&keep_alive.identity()) else {
            return KeepAliveOutcome {
                found: false,
                priority_conflict: false,
            };
        };

        KeepAliveOutcome {
            found: true,
            priority_conflict: entry.observe_priority(keep_alive.priority(), frame),
        }
    }

    /// Leases the sole committed output or reserves the slot for replacement.
    pub(crate) fn lease_output(
        &mut self,
        request: &isolated_layer::CacheRequest,
        key: OutputKey,
        frame: u64,
    ) -> OutputLease<T> {
        let identity = request.identity();

        if !request.has_content_evidence() {
            return self.rejected_lookup(
                identity,
                request.priority(),
                frame,
                OutputMiss::MissingContentEvidence,
                None,
            );
        }

        if request.is_conflicted() {
            return self.rejected_lookup(
                identity,
                request.priority(),
                frame,
                OutputMiss::ContentConflict,
                Some(request.stamp()),
            );
        }

        if key.evidence.is_volatile() {
            return self.rejected_lookup(
                identity,
                request.priority(),
                frame,
                OutputMiss::VolatileInputs,
                None,
            );
        }

        if !request.is_current() {
            let current = request.current_stamp();
            return self.rejected_lookup(
                identity,
                request.priority(),
                frame,
                OutputMiss::ContentChangedBeforeLookup,
                Some(&current),
            );
        }

        let ticket = self.issue_ticket();

        if let Some(entry) = self.entries.get_mut(&identity) {
            let priority_conflict = entry.observe_priority(request.priority(), frame);

            if let Some(active) = &mut entry.active {
                active.contended = true;
                entry.observe_watermarks(request.stamp());

                return OutputLease {
                    target: None,
                    ticket: None,
                    valid: false,
                    cacheable: false,
                    miss: Some(OutputMiss::CompetingWriter),
                    priority_conflict,
                };
            }

            match classify_stamp(&entry.latest_stamp, &entry.watermarks, request.stamp()) {
                StampDisposition::Stale => {
                    return rejected_lease(OutputMiss::StaleContent, priority_conflict);
                }
                StampDisposition::Incomparable => {
                    entry.observe_watermarks(request.stamp());
                    return rejected_lease(OutputMiss::IncomparableContent, priority_conflict);
                }
                StampDisposition::Exact
                | StampDisposition::Advanced
                | StampDisposition::StructuralChange => {
                    entry.accept_stamp(request.stamp());
                }
            }

            if entry
                .committed
                .as_ref()
                .is_some_and(|committed| committed.key.matches(&key))
            {
                let committed = entry.committed.take().expect("matched committed output");
                entry.active = Some(ActiveLease {
                    ticket,
                    frame,
                    expected: key,
                    kind: LeaseKind::Hit,
                    contended: false,
                });
                entry.last_used = frame;

                OutputLease {
                    target: Some(committed.target),
                    ticket: Some(ticket),
                    valid: true,
                    cacheable: true,
                    miss: None,
                    priority_conflict,
                }
            } else {
                let miss = entry
                    .committed
                    .as_ref()
                    .map_or(OutputMiss::Targetless, |committed| {
                        miss_from_key(&committed.key, &key)
                    });
                entry.active = Some(ActiveLease {
                    ticket,
                    frame,
                    expected: key,
                    kind: LeaseKind::Replacement,
                    contended: false,
                });

                OutputLease {
                    target: None,
                    ticket: Some(ticket),
                    valid: false,
                    cacheable: true,
                    miss: Some(miss),
                    priority_conflict,
                }
            }
        } else {
            let _ = self
                .entries
                .insert(identity, Entry::new(request, key, ticket, frame));

            OutputLease {
                target: None,
                ticket: Some(ticket),
                valid: false,
                cacheable: true,
                miss: Some(OutputMiss::UnknownIdentity),
                priority_conflict: false,
            }
        }
    }

    /// Atomically commits a freshly validated output key and target.
    ///
    /// `fresh_key` must be recollected after rendering. Both caller-managed
    /// content generations and exact effect evidence are checked again before
    /// a replacement can displace the old committed output.
    pub(crate) fn store_output(
        &mut self,
        ticket: LeaseTicket,
        request: &isolated_layer::CacheRequest,
        fresh_key: OutputKey,
        frame: u64,
        target: T,
    ) -> StoreOutcome<T> {
        let Some(entry) = self.entries.get_mut(&request.identity()) else {
            return rejected_store(StoreDisposition::UnknownEntry, target);
        };

        let Some(active) = entry.active.as_ref() else {
            return rejected_store(StoreDisposition::TicketMismatch, target);
        };

        if active.ticket != ticket {
            return rejected_store(StoreDisposition::TicketMismatch, target);
        }

        let active = entry.active.take().expect("checked active output lease");
        let current = request.current_stamp();

        let disposition = if active.contended {
            StoreDisposition::CompetingWriter
        } else if active.frame != frame {
            StoreDisposition::FrameMismatch
        } else if !request.has_content_evidence()
            || request.is_conflicted()
            || active.expected.content != *request.stamp()
            || fresh_key.content != *request.stamp()
        {
            StoreDisposition::RequestMismatch
        } else if current != *request.stamp() || current.is_conflicted() || current.is_empty() {
            entry.observe_watermarks(&current);
            StoreDisposition::ContentChanged
        } else if fresh_key.evidence.is_volatile() {
            StoreDisposition::VolatileInputs
        } else {
            match active.expected.first_mismatch(&fresh_key) {
                None => StoreDisposition::Stored,
                Some(KeyMismatch::Content) => StoreDisposition::ContentChanged,
                Some(KeyMismatch::Raster) => StoreDisposition::RasterChanged,
                Some(KeyMismatch::Geometry) => StoreDisposition::GeometryChanged,
                Some(KeyMismatch::EffectInputs) => StoreDisposition::EffectInputsChanged,
            }
        };

        if disposition == StoreDisposition::Stored {
            entry.accept_stamp(request.stamp());
            entry.last_used = frame;
            let replaced = entry.committed.replace(CommittedOutput {
                key: fresh_key,
                target,
            });

            StoreOutcome {
                disposition,
                released: replaced
                    .map(|committed| vec![committed.target])
                    .unwrap_or_default(),
            }
        } else {
            let mut released = Vec::new();

            match active.kind {
                LeaseKind::Hit if entry.committed.is_none() => {
                    entry.committed = Some(CommittedOutput {
                        key: active.expected,
                        target,
                    });
                }
                LeaseKind::Hit | LeaseKind::Replacement => released.push(target),
            }

            StoreOutcome {
                disposition,
                released,
            }
        }
    }

    /// Clears leases left active by an earlier rendered frame.
    pub(crate) fn recover_abandoned(&mut self, frame: u64) -> RecoveryOutcome {
        let mut output_leases = 0usize;

        for entry in self.entries.values_mut() {
            if entry
                .active
                .as_ref()
                .is_some_and(|active| active.frame != frame)
            {
                let _ = entry.active.take();
                output_leases = output_leases.saturating_add(1);
            }
        }

        RecoveryOutcome { output_leases }
    }

    /// Counts current-frame leases which were not returned.
    pub(crate) fn finish_frame(&self, frame: u64) -> LeaseAudit {
        LeaseAudit {
            output_leases: self
                .entries
                .values()
                .filter(|entry| {
                    entry
                        .active
                        .as_ref()
                        .is_some_and(|active| active.frame == frame)
                })
                .count(),
        }
    }

    /// Removes identities not marked within the configured grace period.
    pub(crate) fn sweep(&mut self, frame: u64, grace: u64) -> SweepOutcome<T> {
        let mut released = Vec::new();
        let mut outputs = 0usize;

        self.entries.retain(|_, entry| {
            let remove = entry.active.is_none() && frame.wrapping_sub(entry.marked_frame) > grace;

            if remove {
                outputs = outputs.saturating_add(1);
                if let Some(committed) = entry.committed.take() {
                    released.push(committed.target);
                }
            }

            !remove
        });

        SweepOutcome { released, outputs }
    }

    /// Returns resident bytes split by current output priority.
    pub(crate) fn resident_bytes(&self) -> ResidentBytes {
        let mut bytes = ResidentBytes::default();

        for entry in self.entries.values() {
            let Some(committed) = &entry.committed else {
                continue;
            };
            let target_bytes = committed.target.output_byte_size();

            match entry.priority {
                isolated_layer::CacheResidencyPriority::Normal => {
                    bytes.normal = bytes.normal.saturating_add(target_bytes);
                }
                isolated_layer::CacheResidencyPriority::Protected => {
                    bytes.protected = bytes.protected.saturating_add(target_bytes);
                }
            }
        }

        bytes
    }

    /// Returns all resident target bytes.
    pub(crate) fn bytes(&self) -> u64 {
        self.resident_bytes().total()
    }

    /// Returns output-slot and content-watermark metadata counts.
    pub(crate) fn metadata_counts(&self) -> MetadataCounts {
        MetadataCounts {
            outputs: self.entries.len(),
            resident_outputs: self
                .entries
                .values()
                .filter(|entry| entry.committed.is_some())
                .count(),
            watermark_records: self
                .entries
                .values()
                .map(|entry| entry.watermarks.len())
                .sum(),
        }
    }

    /// Evicts least-recently-used normal targets before protected targets.
    ///
    /// Active slots are skipped. Their targets are either in use by the caller
    /// or are the old committed value protected during replacement.
    pub(crate) fn evict_to_bytes(&mut self, maximum: u64, frame: u64) -> EvictionOutcome<T> {
        let mut remaining_bytes = self.bytes();
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .filter_map(|(&identity, entry)| {
                (entry.active.is_none() && entry.committed.is_some()).then_some(Candidate {
                    identity,
                    tier: eviction_tier(entry.priority),
                    age: frame.wrapping_sub(entry.last_used),
                })
            })
            .collect();

        candidates.sort_by(|left, right| {
            left.tier
                .cmp(&right.tier)
                .then_with(|| right.age.cmp(&left.age))
                .then_with(|| left.identity.cmp(&right.identity))
        });

        let mut evicted = Vec::new();
        for candidate in candidates {
            if remaining_bytes <= maximum {
                break;
            }

            let Some(entry) = self.entries.get_mut(&candidate.identity) else {
                continue;
            };
            let Some(committed) = entry.committed.take() else {
                continue;
            };
            remaining_bytes = remaining_bytes.saturating_sub(committed.target.output_byte_size());
            evicted.push(EvictedTarget {
                target: committed.target,
                tier: candidate.tier,
            });
        }

        EvictionOutcome {
            evicted,
            remaining_bytes,
        }
    }

    fn issue_ticket(&mut self) -> LeaseTicket {
        self.next_ticket = self.next_ticket.wrapping_add(1);
        LeaseTicket(self.next_ticket)
    }

    fn rejected_lookup(
        &mut self,
        identity: u64,
        priority: isolated_layer::CacheResidencyPriority,
        frame: u64,
        miss: OutputMiss,
        watermark: Option<&isolated_layer::ContentStamp>,
    ) -> OutputLease<T> {
        let priority_conflict = self.entries.get_mut(&identity).is_some_and(|entry| {
            if let Some(stamp) = watermark {
                entry.observe_watermarks(stamp);
            }
            entry.observe_priority(priority, frame)
        });

        rejected_lease(miss, priority_conflict)
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    identity: u64,
    tier: EvictionTier,
    age: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StampDisposition {
    Exact,
    Advanced,
    StructuralChange,
    Stale,
    Incomparable,
}

fn classify_stamp(
    latest: &isolated_layer::ContentStamp,
    watermarks: &BTreeMap<u64, u64>,
    incoming: &isolated_layer::ContentStamp,
) -> StampDisposition {
    if latest == incoming {
        return StampDisposition::Exact;
    }

    let structural_change = latest
        .revisions()
        .iter()
        .map(|revision| revision.identity())
        .ne(incoming
            .revisions()
            .iter()
            .map(|revision| revision.identity()));
    let mut advanced = false;
    let mut regressed = false;

    for revision in incoming.revisions() {
        match watermarks.get(&revision.identity()) {
            Some(generation) if revision.generation() < *generation => regressed = true,
            Some(_) | None => {}
        }

        match latest
            .revisions()
            .binary_search_by_key(&revision.identity(), |revision| revision.identity())
            .ok()
            .map(|index| latest.revisions()[index].generation())
        {
            Some(generation) if revision.generation() > generation => advanced = true,
            None => advanced = true,
            Some(_) => {}
        }
    }

    if regressed && (advanced || structural_change) {
        StampDisposition::Incomparable
    } else if regressed {
        StampDisposition::Stale
    } else if structural_change {
        StampDisposition::StructuralChange
    } else if advanced {
        StampDisposition::Advanced
    } else {
        // Unequal normalized stamps with equal identity sets and no generation
        // movement should be impossible, but failing closed is safer.
        StampDisposition::Incomparable
    }
}

fn miss_from_key(committed: &OutputKey, incoming: &OutputKey) -> OutputMiss {
    match committed.first_mismatch(incoming) {
        None => OutputMiss::Targetless,
        Some(KeyMismatch::Content) => OutputMiss::ContentChanged,
        Some(KeyMismatch::Raster) => OutputMiss::RasterChanged,
        Some(KeyMismatch::Geometry) => OutputMiss::GeometryChanged,
        Some(KeyMismatch::EffectInputs) => OutputMiss::EffectInputsChanged,
    }
}

fn rejected_lease<T>(miss: OutputMiss, priority_conflict: bool) -> OutputLease<T> {
    OutputLease {
        target: None,
        ticket: None,
        valid: false,
        cacheable: false,
        miss: Some(miss),
        priority_conflict,
    }
}

fn rejected_store<T>(disposition: StoreDisposition, target: T) -> StoreOutcome<T> {
    StoreOutcome {
        disposition,
        released: vec![target],
    }
}

fn eviction_tier(priority: isolated_layer::CacheResidencyPriority) -> EvictionTier {
    match priority {
        isolated_layer::CacheResidencyPriority::Normal => EvictionTier::Normal,
        isolated_layer::CacheResidencyPriority::Protected => EvictionTier::Protected,
    }
}

fn canonical_bits(value: f32) -> u32 {
    if value == 0.0 {
        0.0f32.to_bits()
    } else {
        value.to_bits()
    }
}

fn rectangle_bits(rectangle: Rectangle) -> [u32; 4] {
    [
        canonical_bits(rectangle.x),
        canonical_bits(rectangle.y),
        canonical_bits(rectangle.width),
        canonical_bits(rectangle.height),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Size, renderer};
    use crate::graphics::Viewport;
    use crate::isolated_layer::CaptureGrid;

    #[derive(Debug, PartialEq, Eq)]
    struct FakeTarget {
        id: u64,
        bytes: u64,
    }

    impl FakeTarget {
        fn new(id: u64) -> Self {
            Self { id, bytes: 16 }
        }
    }

    impl OutputTarget for FakeTarget {
        fn output_byte_size(&self) -> u64 {
            self.bytes
        }
    }

    fn evidence(value: u8) -> effect::LayerInputEvidence {
        let mut inputs = effect::LayerInputRecords::new();
        inputs.record(&value);
        inputs.finish()
    }

    fn context_at(x: f32) -> Context {
        let viewport =
            Viewport::with_physical_size(Size::new(512, 512), renderer::Scale::default());
        let root = Context::root(&viewport, wgpu::TextureFormat::Rgba8Unorm);
        let bounds = Rectangle::new(crate::core::Point::new(x, 20.0), Size::new(80.0, 40.0));
        let mut context = Context::bounded_with_grid(bounds, &root, CaptureGrid::LayerAligned)
            .expect("visible layer context");
        context.set_source_geometry(bounds.size(), bounds);
        context
    }

    fn key(
        request: &isolated_layer::CacheRequest,
        evidence: effect::LayerInputEvidence,
        context: &Context,
        position_sensitive: bool,
    ) -> OutputKey {
        OutputKey::new(
            request,
            evidence,
            context,
            context.represented_bounds,
            position_sensitive,
            7,
        )
    }

    #[test]
    fn one_slot_replaces_x_with_y_and_does_not_retain_x_history() {
        let surface = isolated_layer::SurfaceHandle::new();
        let content = isolated_layer::ContentChangeHandle::new();
        let request = surface.cache_request([&content]);
        let context = context_at(10.0);
        let mut registry = Registry::<FakeTarget>::default();

        let x = key(&request, evidence(1), &context, false);
        let initial = registry.lease_output(&request, x.clone(), 1);
        assert_eq!(initial.miss, Some(OutputMiss::UnknownIdentity));
        let stored = registry.store_output(
            initial.ticket.expect("initial ticket"),
            &request,
            x.clone(),
            1,
            FakeTarget::new(1),
        );
        assert!(stored.stored());

        let y = key(&request, evidence(2), &context, false);
        let replacement = registry.lease_output(&request, y.clone(), 2);
        assert_eq!(replacement.miss, Some(OutputMiss::EffectInputsChanged));
        assert_eq!(registry.bytes(), 16, "old X remains during replacement");
        let stored = registry.store_output(
            replacement.ticket.expect("replacement ticket"),
            &request,
            y,
            2,
            FakeTarget::new(2),
        );
        assert_eq!(stored.released, vec![FakeTarget::new(1)]);

        let back_to_x = registry.lease_output(&request, x, 3);
        assert_eq!(back_to_x.miss, Some(OutputMiss::EffectInputsChanged));
        assert!(back_to_x.target.is_none());
        let stored = registry.store_output(
            back_to_x.ticket.expect("back-to-X ticket"),
            &request,
            key(&request, evidence(1), &context, false),
            3,
            FakeTarget::new(3),
        );
        assert_eq!(stored.released, vec![FakeTarget::new(2)]);
        assert_eq!(registry.metadata_counts().outputs, 1);
        assert_eq!(registry.metadata_counts().resident_outputs, 1);
    }

    #[test]
    fn stale_content_during_lease_rejects_candidate_and_preserves_old_output() {
        let surface = isolated_layer::SurfaceHandle::new();
        let content = isolated_layer::ContentChangeHandle::new();
        let request = surface.cache_request([&content]);
        let context = context_at(10.0);
        let mut registry = Registry::<FakeTarget>::default();
        let x = key(&request, evidence(1), &context, false);

        let initial = registry.lease_output(&request, x.clone(), 1);
        let _ = registry.store_output(
            initial.ticket.expect("initial ticket"),
            &request,
            x,
            1,
            FakeTarget::new(1),
        );

        let y = key(&request, evidence(2), &context, false);
        let replacement = registry.lease_output(&request, y.clone(), 2);
        let _ = content.mark_changed();
        let rejected = registry.store_output(
            replacement.ticket.expect("replacement ticket"),
            &request,
            y,
            2,
            FakeTarget::new(2),
        );

        assert_eq!(rejected.disposition, StoreDisposition::ContentChanged);
        assert_eq!(rejected.released, vec![FakeTarget::new(2)]);
        assert_eq!(registry.bytes(), 16);
        assert_eq!(
            registry
                .entries
                .get(&surface.identity())
                .and_then(|entry| entry.committed.as_ref())
                .map(|committed| committed.target.id),
            Some(1)
        );
    }

    #[test]
    fn competing_writer_invalidates_first_lease_without_displacing_old_output() {
        let surface = isolated_layer::SurfaceHandle::new();
        let content = isolated_layer::ContentChangeHandle::new();
        let request = surface.cache_request([&content]);
        let context = context_at(10.0);
        let mut registry = Registry::<FakeTarget>::default();
        let x = key(&request, evidence(1), &context, false);

        let initial = registry.lease_output(&request, x.clone(), 1);
        let _ = registry.store_output(
            initial.ticket.expect("initial ticket"),
            &request,
            x,
            1,
            FakeTarget::new(1),
        );

        let y = key(&request, evidence(2), &context, false);
        let first = registry.lease_output(&request, y.clone(), 2);
        let second = registry.lease_output(&request, y.clone(), 2);
        assert_eq!(second.miss, Some(OutputMiss::CompetingWriter));
        assert!(!second.cacheable);

        let rejected = registry.store_output(
            first.ticket.expect("first writer ticket"),
            &request,
            y,
            2,
            FakeTarget::new(2),
        );
        assert_eq!(rejected.disposition, StoreDisposition::CompetingWriter);
        assert_eq!(rejected.released, vec![FakeTarget::new(2)]);
        assert_eq!(registry.bytes(), 16);
    }

    #[test]
    fn absolute_geometry_is_conditional_on_translation_sensitivity() {
        let surface = isolated_layer::SurfaceHandle::new();
        let content = isolated_layer::ContentChangeHandle::new();
        let request = surface.cache_request([&content]);
        let first = context_at(10.0);
        let moved = context_at(73.25);

        let invariant_a = key(&request, evidence(1), &first, false);
        let invariant_b = key(&request, evidence(1), &moved, false);
        assert!(invariant_a.matches(&invariant_b));

        let sensitive_a = key(&request, evidence(1), &first, true);
        let sensitive_b = key(&request, evidence(1), &moved, true);
        assert!(!sensitive_a.matches(&sensitive_b));
    }

    #[test]
    fn keep_alive_only_marks_existing_identity_and_normalizes_priority_conflict() {
        let missing = isolated_layer::SurfaceHandle::new();
        let mut registry = Registry::<FakeTarget>::default();
        assert!(!registry.keep_alive(&missing.cache_keep_alive(), 1).found);
        assert_eq!(registry.metadata_counts().outputs, 0);

        let content = isolated_layer::ContentChangeHandle::new();
        let request = missing.cache_request_with(
            isolated_layer::CacheResidencyPriority::Protected,
            [&content],
        );
        let context = context_at(10.0);
        let output_key = key(&request, evidence(1), &context, false);
        let lease = registry.lease_output(&request, output_key.clone(), 2);
        let _ = registry.store_output(
            lease.ticket.expect("ticket"),
            &request,
            output_key,
            2,
            FakeTarget::new(1),
        );

        let outcome = registry.keep_alive(&missing.cache_keep_alive(), 2);
        assert!(outcome.found);
        assert!(outcome.priority_conflict);
        assert_eq!(registry.resident_bytes().normal, 16);
        assert_eq!(registry.resident_bytes().protected, 0);
    }

    #[test]
    fn recovery_sweep_and_budget_operate_on_whole_output_slots() {
        let first_surface = isolated_layer::SurfaceHandle::new();
        let second_surface = isolated_layer::SurfaceHandle::new();
        let first_content = isolated_layer::ContentChangeHandle::new();
        let second_content = isolated_layer::ContentChangeHandle::new();
        let first_request = first_surface.cache_request([&first_content]);
        let second_request = second_surface.cache_request_with(
            isolated_layer::CacheResidencyPriority::Protected,
            [&second_content],
        );
        let context = context_at(10.0);
        let mut registry = Registry::<FakeTarget>::default();

        let first_key = key(&first_request, evidence(1), &context, false);
        let first_lease = registry.lease_output(&first_request, first_key.clone(), 1);
        let _ = registry.store_output(
            first_lease.ticket.expect("first ticket"),
            &first_request,
            first_key,
            1,
            FakeTarget::new(1),
        );
        let second_key = key(&second_request, evidence(2), &context, false);
        let second_lease = registry.lease_output(&second_request, second_key.clone(), 2);
        let _ = registry.store_output(
            second_lease.ticket.expect("second ticket"),
            &second_request,
            second_key.clone(),
            2,
            FakeTarget::new(2),
        );

        let abandoned = registry.lease_output(&second_request, second_key, 3);
        assert!(abandoned.valid);
        assert_eq!(registry.finish_frame(3).output_leases, 1);
        assert_eq!(registry.recover_abandoned(4).output_leases, 1);

        let eviction = registry.evict_to_bytes(0, 4);
        assert_eq!(eviction.evicted.len(), 1);
        assert_eq!(eviction.evicted[0].tier, EvictionTier::Normal);
        assert_eq!(eviction.evicted[0].target.id, 1);
        assert_eq!(eviction.remaining_bytes, 0);

        let swept = registry.sweep(10, 1);
        assert_eq!(swept.outputs, 2);
        assert!(swept.released.is_empty());
        assert_eq!(registry.metadata_counts(), MetadataCounts::default());
    }

    #[test]
    fn empty_content_evidence_fails_closed_without_creating_a_slot() {
        let surface = isolated_layer::SurfaceHandle::new();
        let request = surface.cache_request([]);
        let context = context_at(10.0);
        let mut registry = Registry::<FakeTarget>::default();
        let lease = registry.lease_output(&request, key(&request, evidence(1), &context, false), 1);

        assert_eq!(lease.miss, Some(OutputMiss::MissingContentEvidence));
        assert!(!lease.cacheable);
        assert_eq!(registry.metadata_counts(), MetadataCounts::default());
    }

    #[test]
    fn fresh_request_is_accepted_after_pre_lookup_generation_race() {
        let surface = isolated_layer::SurfaceHandle::new();
        let content = isolated_layer::ContentChangeHandle::new();
        let initial_request = surface.cache_request([&content]);
        let context = context_at(10.0);
        let mut registry = Registry::<FakeTarget>::default();
        let initial_key = key(&initial_request, evidence(1), &context, false);
        let initial = registry.lease_output(&initial_request, initial_key.clone(), 1);
        let _ = registry.store_output(
            initial.ticket.expect("initial ticket"),
            &initial_request,
            initial_key,
            1,
            FakeTarget::new(1),
        );

        let stale_request = surface.cache_request([&content]);
        let _ = content.mark_changed();
        let rejected = registry.lease_output(
            &stale_request,
            key(&stale_request, evidence(1), &context, false),
            2,
        );
        assert_eq!(rejected.miss, Some(OutputMiss::ContentChangedBeforeLookup));

        let fresh_request = stale_request.resnapshot();
        let accepted = registry.lease_output(
            &fresh_request,
            key(&fresh_request, evidence(1), &context, false),
            2,
        );
        assert_eq!(accepted.miss, Some(OutputMiss::ContentChanged));
        assert!(accepted.cacheable);
    }
}
