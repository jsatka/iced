//! Bounded layer-isolated drawing.

mod composite;
mod context;
pub mod effect;
mod pool;
mod recording;
mod retained;

pub(crate) use composite::{
    Prepared as PreparedComposite, Storage as CompositeStorage, render as render_composite,
    render_backdrop,
};
pub(crate) use context::{CaptureGrid, Context, Placement};
pub use effect::{
    Context as EffectContext, Effect, EffectStack, Layer, LayerEffect, LayerInputEvidence,
    LayerInputRecords, Pipeline, PipelineRegistry, Renderer, Requirements, TextureViews,
};
pub(crate) use effect::{Storage as LayerEffectStorage, context as effect_context};
pub(crate) use pool::{Pool, Target};
pub(crate) use recording::{Leaf, Node, PreparedLayer, Recorder, Sequence};
pub(crate) use retained::{
    EvictionTier, LeaseTicket, OutputKey, OutputMiss, Registry, StoreDisposition, StoreOutcome,
};

use crate::core::isolated_layer::{CacheKeepAlive, CacheRequest, CacheResidencyPriority};

use std::cell::RefCell;
use std::collections::HashMap;

pub(crate) struct PreparedIsolatedLayer {
    pub context: Context,
    pub targets: Vec<Target>,
    pub composite: PreparedComposite,
    pub output_lease: Option<(CacheRequest, OutputKey, LeaseTicket)>,
    pub output_valid: bool,
}

pub(crate) struct PreparedLayerEffect {
    pub context: Context,
    pub targets: Vec<Target>,
    pub backdrop: Option<usize>,
    pub composite: PreparedComposite,
    pub passes: Vec<PreparedEffectPass>,
    pub output: usize,
    pub output_lease: Option<(CacheRequest, OutputKey, LeaseTicket)>,
    pub output_valid: bool,
}

pub(crate) struct PreparedEffectPass {
    pub effect: usize,
    pub pass: usize,
    pub stage_input: usize,
    pub previous: usize,
    pub output: usize,
    pub uses_backdrop: bool,
    pub writes_every_pixel: bool,
    pub prepared: Box<dyn effect::Erased>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannedEffectPass {
    pub effect: usize,
    pub pass: usize,
    pub stage_input: usize,
    pub previous: usize,
    pub output: usize,
    pub uses_backdrop: bool,
    pub writes_every_pixel: bool,
}

pub(crate) struct EffectPassPlan {
    pub passes: Vec<PlannedEffectPass>,
    pub backdrop: Option<usize>,
    pub output: usize,
    pub target_count: usize,
}

/// Plans the index-only portion of the shared-final-canvas effect chain.
///
/// Target zero is always the captured child. Every pass receives a dedicated
/// output target, while each stage keeps one stable input: the captured child
/// for the first stage and the preceding stage's final output thereafter.
pub(crate) fn plan_effect_passes(effects: &EffectStack) -> EffectPassPlan {
    let stage_requirements: Vec<_> = effects.stage_requirements().collect();
    let backdrop = stage_requirements
        .iter()
        .any(|requirements| requirements.needs_backdrop())
        .then_some(1);
    let mut next_output = 1 + usize::from(backdrop.is_some());
    let mut current_output = 0;
    let mut passes = Vec::new();

    for (effect, requirements) in stage_requirements.into_iter().enumerate() {
        let stage_input = current_output;

        for pass in 0..requirements.pass_count() {
            let output = next_output;
            next_output += 1;
            let previous = if pass == 0 { stage_input } else { output - 1 };

            passes.push(PlannedEffectPass {
                effect,
                pass,
                stage_input,
                previous,
                output,
                uses_backdrop: requirements.needs_backdrop(),
                writes_every_pixel: requirements.fully_overwrites(),
            });

            current_output = output;
        }
    }

    EffectPassPlan {
        passes,
        backdrop,
        output: current_output,
        target_count: next_output,
    }
}

/// Renderer-local limits for retained and transient GPU textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum bytes owned by the free target pool and retained registry together.
    pub budget_bytes: u64,
    /// Number of completed rendered frames an unmarked output may survive.
    pub grace_frames: u64,
}

impl Limits {
    /// Default native renderer budget (128 MiB).
    pub const NATIVE_BUDGET_BYTES: u64 = 128 * 1024 * 1024;
    /// Default WebAssembly renderer budget (32 MiB).
    pub const WASM_BUDGET_BYTES: u64 = 32 * 1024 * 1024;
    /// Default rendered-frame grace period.
    pub const GRACE_FRAMES: u64 = 2;

    /// Creates renderer-local isolated-layer limits.
    pub const fn new(budget_bytes: u64, grace_frames: u64) -> Self {
        Self {
            budget_bytes,
            grace_frames,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        #[cfg(target_arch = "wasm32")]
        let budget_bytes = Self::WASM_BUDGET_BYTES;
        #[cfg(not(target_arch = "wasm32"))]
        let budget_bytes = Self::NATIVE_BUDGET_BYTES;

        Self::new(budget_bytes, Self::GRACE_FRAMES)
    }
}

/// Per-frame isolated-layer rendering diagnostics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Diagnostics {
    /// Whether the most recent frame used segmented recording.
    pub segmented: bool,
    /// Number of visible isolated-layer nodes.
    pub nodes: usize,
    /// Maximum visible isolated-layer nesting depth.
    pub max_depth: usize,
    /// Renderer-owned targets allocated during the frame.
    pub allocations: usize,
    /// Renderer-owned targets reused from the transient pool.
    pub pool_hits: usize,
    /// Effect passes encoded during the frame.
    pub isolated_layer_passes: usize,
    /// Whether a root intermediate was activated for backdrop sampling.
    pub root_intermediate: bool,
    /// Final-output cache hits.
    pub output_cache_hits: usize,
    /// Final-output cache misses and fail-closed bypasses.
    pub output_cache_misses: usize,
    /// Cache requests bypassed because an effect needs the parent backdrop.
    pub output_cache_bypass_backdrop: usize,
    /// Cache requests bypassed because no child-content evidence was supplied.
    pub output_cache_bypass_missing_content: usize,
    /// Cache requests bypassed because effect evidence was volatile.
    pub output_cache_bypass_volatile: usize,
    /// Misses for a previously unseen output identity.
    pub output_cache_miss_unknown_identity: usize,
    /// Misses for known output metadata without resident pixels.
    pub output_cache_miss_targetless: usize,
    /// Misses caused by conflicting child-content observations.
    pub output_cache_miss_content_conflict: usize,
    /// Misses because content changed before the late lookup.
    pub output_cache_miss_content_changed_before_lookup: usize,
    /// Misses caused by an observation older than the slot watermark.
    pub output_cache_miss_stale_content: usize,
    /// Misses caused by both advancing and regressing content observations.
    pub output_cache_miss_incomparable_content: usize,
    /// Misses caused by changed child-content evidence.
    pub output_cache_miss_content_changed: usize,
    /// Misses caused by changed rasterization facts.
    pub output_cache_miss_raster_changed: usize,
    /// Misses caused by absolute movement of a position-sensitive output.
    pub output_cache_miss_geometry_changed: usize,
    /// Misses caused by changed exact effect or stack evidence.
    pub output_cache_miss_effect_inputs_changed: usize,
    /// Misses caused by concurrent producers for one output identity.
    pub output_cache_miss_competing_writer: usize,
    /// Same-frame observations which disagreed on residency priority.
    pub residency_priority_conflicts: usize,
    /// Rendered candidates rejected during store-time validation.
    pub rejected_output_stores: usize,
    /// Rejected stores whose output metadata no longer exists.
    pub rejected_store_unknown_entry: usize,
    /// Rejected stores presenting a non-owning ticket.
    pub rejected_store_ticket_mismatch: usize,
    /// Rejected stores returned in a different rendered frame.
    pub rejected_store_frame_mismatch: usize,
    /// Rejected stores whose slot had a competing producer.
    pub rejected_store_competing_writer: usize,
    /// Rejected stores whose request did not match the lease.
    pub rejected_store_request_mismatch: usize,
    /// Rejected stores because child content changed during rendering.
    pub rejected_store_content_changed: usize,
    /// Rejected stores because rasterization facts changed.
    pub rejected_store_raster_changed: usize,
    /// Rejected stores because position-sensitive geometry changed.
    pub rejected_store_geometry_changed: usize,
    /// Rejected stores because exact effect inputs changed.
    pub rejected_store_effect_inputs_changed: usize,
    /// Rejected stores because recollected inputs were volatile.
    pub rejected_store_volatile_inputs: usize,
    /// Unique identity-only keep-alives consumed by this frame.
    pub pending_keep_alive_identities: usize,
    /// Output entries removed by the post-frame liveness sweep.
    pub sweep_output_removals: usize,
    /// Output leases recovered at the next rendered-frame boundary.
    pub recovered_abandoned_output_leases: usize,
    /// Current-frame output leases still active at the finish-frame audit.
    pub unfinished_output_leases: usize,
    /// Free targets discarded by the idle-age trim.
    pub pool_idle_trims: usize,
    /// Free targets discarded to satisfy the renderer budget.
    pub pool_budget_trims: usize,
    /// Normal-priority outputs evicted to satisfy the renderer budget.
    pub budget_evictions_normal: usize,
    /// Protected-priority outputs evicted to satisfy the renderer budget.
    pub budget_evictions_protected: usize,
    /// Renderer-owned free transient bytes after the most recent draw.
    pub pool_bytes: u64,
    /// Renderer-owned resident output-cache bytes after the most recent draw.
    pub cache_bytes: u64,
    /// Resident normal-priority output bytes.
    pub normal_priority_bytes: u64,
    /// Resident protected-priority output bytes.
    pub protected_priority_bytes: u64,
    /// Free-pool plus retained-registry bytes after the most recent draw.
    pub total_gpu_texture_bytes: u64,
    /// Output-slot metadata entries after the most recent draw.
    pub output_entries: usize,
    /// Output slots with a resident committed target.
    pub resident_output_entries: usize,
    /// Monotonic child-content watermark records.
    pub watermark_records: usize,
    /// Targets evicted by liveness, age, or budget during the frame.
    pub evictions: usize,
}

#[derive(Debug, Clone)]
struct PendingKeepAlive {
    request: CacheKeepAlive,
    priority_conflicted: bool,
}

impl PendingKeepAlive {
    fn new(request: CacheKeepAlive) -> Self {
        Self {
            request,
            priority_conflicted: false,
        }
    }

    fn merge(&mut self, incoming: CacheKeepAlive) {
        if self.request.priority() != incoming.priority() {
            if incoming.priority() == CacheResidencyPriority::Normal {
                self.request = incoming;
            }
            self.priority_conflicted = true;
        }
    }
}

pub(crate) struct State {
    pub pool: Pool,
    pub registry: Registry,
    pub diagnostics: Diagnostics,
    pub frame: u64,
    limits: Limits,
    pending_keep_alives: RefCell<HashMap<u64, PendingKeepAlive>>,
    frame_keep_alives: HashMap<u64, PendingKeepAlive>,
}

impl Default for State {
    fn default() -> Self {
        Self::with_limits(Limits::default())
    }
}

impl State {
    pub(crate) fn with_limits(limits: Limits) -> Self {
        Self {
            pool: Pool::default(),
            registry: Registry::default(),
            diagnostics: Diagnostics::default(),
            frame: 0,
            limits,
            pending_keep_alives: RefCell::default(),
            frame_keep_alives: HashMap::new(),
        }
    }

    pub(crate) fn limits(&self) -> Limits {
        self.limits
    }

    pub(crate) fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    /// Adds an identity-only keep-alive to the bounded pending sink.
    pub(crate) fn mark_cache_alive(&self, keep_alive: CacheKeepAlive) {
        let identity = keep_alive.identity();
        let mut pending = self.pending_keep_alives.borrow_mut();

        match pending.entry(identity) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().merge(keep_alive);
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let _ = entry.insert(PendingKeepAlive::new(keep_alive));
            }
        }
    }

    /// Replaces the next-frame snapshot with keep-alives collected since the previous reset.
    pub(crate) fn snapshot_pending_keep_alives(&mut self) {
        self.frame_keep_alives = std::mem::take(self.pending_keep_alives.get_mut());
    }

    pub(crate) fn begin_frame(&mut self, segmented: bool) {
        self.frame = self.frame.wrapping_add(1);
        self.diagnostics = Diagnostics {
            segmented,
            ..Diagnostics::default()
        };

        let recovered = self.registry.recover_abandoned(self.frame);
        self.diagnostics.recovered_abandoned_output_leases = recovered.output_leases;

        let keep_alives = std::mem::take(&mut self.frame_keep_alives);
        self.diagnostics.pending_keep_alive_identities = keep_alives.len();

        for pending in keep_alives.into_values() {
            let outcome = self.registry.keep_alive(&pending.request, self.frame);
            if pending.priority_conflicted || outcome.priority_conflict {
                self.diagnostics.residency_priority_conflicts = self
                    .diagnostics
                    .residency_priority_conflicts
                    .saturating_add(1);
            }
        }
    }

    pub(crate) fn finish_frame(&mut self) {
        let audit = self.registry.finish_frame(self.frame);
        self.diagnostics.unfinished_output_leases = audit.output_leases;

        let swept = self.registry.sweep(self.frame, self.limits.grace_frames);
        self.diagnostics.sweep_output_removals = swept.outputs;
        self.diagnostics.evictions = self
            .diagnostics
            .evictions
            .saturating_add(swept.released.len());
        self.release_targets(swept.released);

        self.enforce_budget();
        self.refresh_usage();

        debug_assert!(
            self.diagnostics.total_gpu_texture_bytes <= self.limits.budget_bytes,
            "GPU target ownership exceeds the renderer-local budget: {} > {}",
            self.diagnostics.total_gpu_texture_bytes,
            self.limits.budget_bytes,
        );
    }

    pub(crate) fn release_targets(&mut self, targets: Vec<Target>) {
        for target in targets {
            self.pool.release(target, self.frame);
        }
    }

    pub(crate) fn record_output_miss(&mut self, miss: OutputMiss) {
        self.diagnostics.output_cache_misses =
            self.diagnostics.output_cache_misses.saturating_add(1);

        let counter = match miss {
            OutputMiss::UnknownIdentity => &mut self.diagnostics.output_cache_miss_unknown_identity,
            OutputMiss::Targetless => &mut self.diagnostics.output_cache_miss_targetless,
            OutputMiss::MissingContentEvidence => {
                &mut self.diagnostics.output_cache_bypass_missing_content
            }
            OutputMiss::ContentConflict => &mut self.diagnostics.output_cache_miss_content_conflict,
            OutputMiss::ContentChangedBeforeLookup => {
                &mut self
                    .diagnostics
                    .output_cache_miss_content_changed_before_lookup
            }
            OutputMiss::StaleContent => &mut self.diagnostics.output_cache_miss_stale_content,
            OutputMiss::IncomparableContent => {
                &mut self.diagnostics.output_cache_miss_incomparable_content
            }
            OutputMiss::ContentChanged => &mut self.diagnostics.output_cache_miss_content_changed,
            OutputMiss::RasterChanged => &mut self.diagnostics.output_cache_miss_raster_changed,
            OutputMiss::GeometryChanged => &mut self.diagnostics.output_cache_miss_geometry_changed,
            OutputMiss::EffectInputsChanged => {
                &mut self.diagnostics.output_cache_miss_effect_inputs_changed
            }
            OutputMiss::VolatileInputs => &mut self.diagnostics.output_cache_bypass_volatile,
            OutputMiss::CompetingWriter => &mut self.diagnostics.output_cache_miss_competing_writer,
        };
        *counter = counter.saturating_add(1);
    }

    pub(crate) fn record_store(&mut self, outcome: StoreOutcome) -> StoreDisposition {
        let stored = outcome.stored();
        let StoreOutcome {
            disposition,
            released,
        } = outcome;

        if !stored {
            self.diagnostics.rejected_output_stores =
                self.diagnostics.rejected_output_stores.saturating_add(1);

            let counter = match disposition {
                StoreDisposition::Stored => unreachable!("stored outcome handled above"),
                StoreDisposition::UnknownEntry => {
                    &mut self.diagnostics.rejected_store_unknown_entry
                }
                StoreDisposition::TicketMismatch => {
                    &mut self.diagnostics.rejected_store_ticket_mismatch
                }
                StoreDisposition::FrameMismatch => {
                    &mut self.diagnostics.rejected_store_frame_mismatch
                }
                StoreDisposition::CompetingWriter => {
                    &mut self.diagnostics.rejected_store_competing_writer
                }
                StoreDisposition::RequestMismatch => {
                    &mut self.diagnostics.rejected_store_request_mismatch
                }
                StoreDisposition::ContentChanged => {
                    &mut self.diagnostics.rejected_store_content_changed
                }
                StoreDisposition::RasterChanged => {
                    &mut self.diagnostics.rejected_store_raster_changed
                }
                StoreDisposition::GeometryChanged => {
                    &mut self.diagnostics.rejected_store_geometry_changed
                }
                StoreDisposition::EffectInputsChanged => {
                    &mut self.diagnostics.rejected_store_effect_inputs_changed
                }
                StoreDisposition::VolatileInputs => {
                    &mut self.diagnostics.rejected_store_volatile_inputs
                }
            };
            *counter = counter.saturating_add(1);
        }

        self.release_targets(released);
        disposition
    }

    fn enforce_budget(&mut self) {
        let idle_trims = self.pool.trim_idle(self.frame);
        self.diagnostics.pool_idle_trims =
            self.diagnostics.pool_idle_trims.saturating_add(idle_trims);
        self.diagnostics.evictions = self.diagnostics.evictions.saturating_add(idle_trims);

        let registry_bytes = self.registry.bytes();
        let pool_trims = self.pool.trim_to_bytes(
            self.limits.budget_bytes.saturating_sub(registry_bytes),
            self.frame,
        );
        self.diagnostics.pool_budget_trims = self
            .diagnostics
            .pool_budget_trims
            .saturating_add(pool_trims);
        self.diagnostics.evictions = self.diagnostics.evictions.saturating_add(pool_trims);

        let pool_bytes = self.pool.bytes();
        let outcome = self.registry.evict_to_bytes(
            self.limits.budget_bytes.saturating_sub(pool_bytes),
            self.frame,
        );

        for evicted in outcome.evicted {
            match evicted.tier {
                EvictionTier::Normal => {
                    self.diagnostics.budget_evictions_normal =
                        self.diagnostics.budget_evictions_normal.saturating_add(1);
                }
                EvictionTier::Protected => {
                    self.diagnostics.budget_evictions_protected = self
                        .diagnostics
                        .budget_evictions_protected
                        .saturating_add(1);
                }
            }
            self.diagnostics.evictions = self.diagnostics.evictions.saturating_add(1);
            drop(evicted.target);
        }

        debug_assert_eq!(outcome.remaining_bytes, self.registry.bytes());
    }

    fn refresh_usage(&mut self) {
        let resident = self.registry.resident_bytes();
        let metadata = self.registry.metadata_counts();
        let pool_bytes = self.pool.bytes();

        self.diagnostics.pool_bytes = pool_bytes;
        self.diagnostics.normal_priority_bytes = resident.normal;
        self.diagnostics.protected_priority_bytes = resident.protected;
        self.diagnostics.cache_bytes = resident.total();
        self.diagnostics.total_gpu_texture_bytes =
            pool_bytes.saturating_add(self.diagnostics.cache_bytes);
        self.diagnostics.output_entries = metadata.outputs;
        self.diagnostics.resident_output_entries = metadata.resident_outputs;
        self.diagnostics.watermark_records = metadata.watermark_records;

        debug_assert_eq!(self.diagnostics.cache_bytes, self.registry.bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::isolated_layer::SurfaceHandle;

    #[derive(Debug, Clone, PartialEq)]
    struct PlannedStage(effect::Requirements);

    impl effect::LayerEffect for PlannedStage {
        type PreparedPass = ();

        fn requirements(&self) -> effect::Requirements {
            self.0
        }

        fn prepare_pass(
            &self,
            _pipelines: &mut effect::PipelineRegistry<'_>,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _pass: usize,
            _context: &effect::Context,
            _views: effect::TextureViews<'_>,
        ) {
        }

        fn encode_pass(
            &self,
            _pipelines: &effect::PipelineRegistry<'_>,
            _prepared: &Self::PreparedPass,
            _encoder: &mut wgpu::CommandEncoder,
            _pass: usize,
            _context: &effect::Context,
            _views: effect::TextureViews<'_>,
        ) {
        }
    }

    #[test]
    fn production_pass_plan_chains_stage_outputs_and_scopes_backdrop_per_stage() {
        let effects = EffectStack::new()
            .with(PlannedStage(effect::Requirements::passes(2)))
            .with(PlannedStage(
                effect::Requirements::passes(1)
                    .with_backdrop()
                    .writes_every_pixel(),
            ))
            .with(PlannedStage(
                effect::Requirements::passes(3).writes_every_pixel(),
            ));

        let plan = plan_effect_passes(&effects);

        assert_eq!(plan.backdrop, Some(1));
        assert_eq!(plan.output, 7);
        assert_eq!(plan.target_count, 8);
        assert_eq!(
            plan.passes,
            vec![
                PlannedEffectPass {
                    effect: 0,
                    pass: 0,
                    stage_input: 0,
                    previous: 0,
                    output: 2,
                    uses_backdrop: false,
                    writes_every_pixel: false,
                },
                PlannedEffectPass {
                    effect: 0,
                    pass: 1,
                    stage_input: 0,
                    previous: 2,
                    output: 3,
                    uses_backdrop: false,
                    writes_every_pixel: false,
                },
                PlannedEffectPass {
                    effect: 1,
                    pass: 0,
                    stage_input: 3,
                    previous: 3,
                    output: 4,
                    uses_backdrop: true,
                    writes_every_pixel: true,
                },
                PlannedEffectPass {
                    effect: 2,
                    pass: 0,
                    stage_input: 4,
                    previous: 4,
                    output: 5,
                    uses_backdrop: false,
                    writes_every_pixel: true,
                },
                PlannedEffectPass {
                    effect: 2,
                    pass: 1,
                    stage_input: 4,
                    previous: 5,
                    output: 6,
                    uses_backdrop: false,
                    writes_every_pixel: true,
                },
                PlannedEffectPass {
                    effect: 2,
                    pass: 2,
                    stage_input: 4,
                    previous: 6,
                    output: 7,
                    uses_backdrop: false,
                    writes_every_pixel: true,
                },
            ]
        );
    }

    #[test]
    fn production_pass_plan_preserves_counts_above_the_legacy_limit() {
        let effects = EffectStack::new()
            .with(PlannedStage(effect::Requirements::passes(32)))
            .with(PlannedStage(effect::Requirements::passes(17)));

        let plan = plan_effect_passes(&effects);

        assert_eq!(plan.passes.len(), 49);
        assert_eq!(plan.passes[31].effect, 0);
        assert_eq!(plan.passes[31].pass, 31);
        assert_eq!(plan.passes[32].effect, 1);
        assert_eq!(plan.passes[32].pass, 0);
        assert_eq!(plan.passes[32].stage_input, plan.passes[31].output);
        assert_eq!(plan.passes[48].effect, 1);
        assert_eq!(plan.passes[48].pass, 16);
        assert_eq!(plan.output, plan.passes[48].output);
    }

    #[test]
    fn empty_stack_uses_the_captured_child_as_its_output() {
        let plan = plan_effect_passes(&EffectStack::new());

        assert!(plan.passes.is_empty());
        assert_eq!(plan.backdrop, None);
        assert_eq!(plan.output, 0);
        assert_eq!(plan.target_count, 1);
    }

    #[test]
    fn default_limits_are_platform_specific_and_use_two_frame_grace() {
        let limits = Limits::default();

        #[cfg(target_arch = "wasm32")]
        assert_eq!(limits.budget_bytes, Limits::WASM_BUDGET_BYTES);
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(limits.budget_bytes, Limits::NATIVE_BUDGET_BYTES);

        assert_eq!(limits.grace_frames, 2);
    }

    #[test]
    fn pending_keep_alives_are_identity_bounded_and_priority_conflicts_fail_normal() {
        let surface = SurfaceHandle::new();
        let state = State::default();

        state.mark_cache_alive(surface.cache_keep_alive_with(CacheResidencyPriority::Protected));
        state.mark_cache_alive(surface.cache_keep_alive());

        let pending = state.pending_keep_alives.borrow();
        assert_eq!(pending.len(), 1);
        let observation = pending.get(&surface.identity()).expect("output keep-alive");
        assert_eq!(
            observation.request.priority(),
            CacheResidencyPriority::Normal
        );
        assert!(observation.priority_conflicted);
    }

    #[test]
    fn reset_replaces_the_keep_alive_snapshot_and_begin_consumes_it_once() {
        let stale = SurfaceHandle::new();
        let current = SurfaceHandle::new();
        let mut state = State::default();

        state.mark_cache_alive(stale.cache_keep_alive());
        state.snapshot_pending_keep_alives();
        assert!(state.pending_keep_alives.get_mut().is_empty());
        assert!(state.frame_keep_alives.contains_key(&stale.identity()));

        state.mark_cache_alive(current.cache_keep_alive());
        state.snapshot_pending_keep_alives();
        assert_eq!(state.frame_keep_alives.len(), 1);
        assert!(!state.frame_keep_alives.contains_key(&stale.identity()));
        assert!(state.frame_keep_alives.contains_key(&current.identity()));

        state.begin_frame(true);
        assert!(state.frame_keep_alives.is_empty());
        assert_eq!(state.diagnostics.pending_keep_alive_identities, 1);
        assert!(state.diagnostics.segmented);

        state.begin_frame(false);
        assert_eq!(state.diagnostics.pending_keep_alive_identities, 0);
    }

    #[test]
    fn diagnostic_helpers_use_output_only_taxonomy() {
        let mut state = State::with_limits(Limits::new(17, 3));

        state.record_output_miss(OutputMiss::EffectInputsChanged);
        let disposition = state.record_store(StoreOutcome {
            disposition: StoreDisposition::TicketMismatch,
            released: Vec::new(),
        });

        assert_eq!(state.limits(), Limits::new(17, 3));
        assert_eq!(state.diagnostics.output_cache_misses, 1);
        assert_eq!(state.diagnostics.output_cache_miss_effect_inputs_changed, 1);
        assert_eq!(disposition, StoreDisposition::TicketMismatch);
        assert_eq!(state.diagnostics.rejected_output_stores, 1);
        assert_eq!(state.diagnostics.rejected_store_ticket_mismatch, 1);
    }
}
