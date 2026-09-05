//! Layer-isolated drawing and retained surfaces.

use crate::Rectangle;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod cache;

pub use cache::{CacheKeepAlive, CacheRequest, CacheResidencyPriority, ContentStamp, Revision};

/// Fixed-function composition settings for an isolated layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Composite {
    opacity: f32,
    blend_mode: BlendMode,
    positioning: CompositePositioning,
}

impl Composite {
    /// Creates source-over composition at the given group opacity.
    pub fn source_over(opacity: f32) -> Self {
        Self {
            opacity: canonical(opacity, 0.0, 1.0),
            blend_mode: BlendMode::SourceOver,
            positioning: CompositePositioning::default(),
        }
    }

    /// Creates additive composition at the given group opacity.
    pub fn additive(opacity: f32) -> Self {
        Self {
            opacity: canonical(opacity, 0.0, 1.0),
            blend_mode: BlendMode::Add,
            positioning: CompositePositioning::default(),
        }
    }

    /// Sets the opacity applied to the whole captured group.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = canonical(opacity, 0.0, 1.0);
        self
    }

    /// Sets the fixed-function blend mode.
    pub fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    /// Sets how the completed layer is positioned in its parent.
    pub fn with_positioning(mut self, positioning: CompositePositioning) -> Self {
        self.positioning = positioning;
        self
    }

    /// Returns the opacity applied to the whole captured group.
    pub fn opacity(self) -> f32 {
        self.opacity
    }

    /// Returns the fixed-function blend mode.
    pub fn blend_mode(self) -> BlendMode {
        self.blend_mode
    }

    /// Returns how the completed layer is positioned in its parent.
    pub fn positioning(self) -> CompositePositioning {
        self.positioning
    }
}

impl Default for Composite {
    fn default() -> Self {
        Self::source_over(1.0)
    }
}

/// Requested positioning of a completed isolated layer in its immediate parent.
///
/// This setting affects final composition only. It does not participate in retained-output
/// validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompositePositioning {
    /// Requests pixel-snapped final composition.
    ///
    /// This is the compatibility default. Transient captures may still preserve a fractional
    /// raster phase internally.
    #[default]
    Snapped,
    /// Requests final composition at the exact physical origin.
    ///
    /// The WGPU renderer uses linear reconstruction. This can make high-contrast content,
    /// including text, slightly softer while moving. Renderers which draw isolated content
    /// directly may not need a separate reconstruction step.
    Subpixel,
}

/// A portable fixed-function blend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Premultiplied source-over blending.
    SourceOver,
    /// Additive blending.
    Add,
}

/// An effect-free isolated-layer request.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// The capture and output bounds.
    ///
    /// They use the same coordinate system as primitives recorded inside the
    /// layer scope.
    pub bounds: Rectangle,
    /// The final composite clip, expressed in the same coordinate system as
    /// [`Self::bounds`].
    ///
    /// This clip does not restrict source capture. A layer effect may read captured pixels
    /// inside [`bounds`](Self::bounds) but outside this rectangle before the result is clipped
    /// during composition.
    pub clip: Rectangle,
    /// How the resulting pixels are composed into the parent.
    pub composite: Composite,
    /// Whether captured child pixels depend on absolute translation.
    pub content_depends_on_translation: bool,
    /// Optional final-output cache identity, content evidence, and residency priority.
    pub output_cache_request: Option<CacheRequest>,
}

impl Layer {
    /// Creates a layer request with source-over composition.
    pub fn new(bounds: Rectangle, clip: Rectangle) -> Self {
        Self {
            bounds,
            clip,
            composite: Composite::default(),
            content_depends_on_translation: false,
            output_cache_request: None,
        }
    }

    /// Sets composition settings.
    pub fn composite(mut self, composite: Composite) -> Self {
        self.composite = composite;
        self
    }

    /// Declares whether captured child pixels depend on absolute translation.
    ///
    /// Child content is considered translation invariant by default.
    pub fn content_depends_on_translation(mut self, depends: bool) -> Self {
        self.content_depends_on_translation = depends;
        self
    }

    /// Requests caching of the final pre-composite output under the logical `surface` identity.
    ///
    /// `content` supplies caller-managed evidence for every captured-content input that may
    /// change the output pixels. A request with no observations is retained in the layer value,
    /// but renderers must fail closed and bypass output reuse.
    pub fn cache_output<'a>(
        self,
        surface: &SurfaceHandle,
        content: impl IntoIterator<Item = &'a ContentChangeHandle>,
    ) -> Self {
        self.cache_output_with(surface, CacheResidencyPriority::Normal, content)
    }

    /// Requests final-output caching with an explicit residency priority.
    pub fn cache_output_with<'a>(
        mut self,
        surface: &SurfaceHandle,
        priority: CacheResidencyPriority,
        content: impl IntoIterator<Item = &'a ContentChangeHandle>,
    ) -> Self {
        self.output_cache_request = Some(surface.cache_request_with(priority, content));
        self
    }
}

/// Residency authority for one backend-agnostic retained layer-output slot.
///
/// Reuse a handle across rebuilds or moves of the same logical producer. Distinct producers
/// should use distinct handles. This identity says where an output may be retained; it is not
/// evidence that the pixels are unchanged. Supply separate [`ContentChangeHandle`] observations
/// when creating a cache request.
///
/// Holding or cloning a handle does not keep GPU pixels resident, and retained pixels are scoped
/// to each renderer that observes the producer.
#[derive(Clone)]
pub struct SurfaceHandle {
    identity: u64,
}

impl SurfaceHandle {
    /// Creates a new logical output-slot identity.
    pub fn new() -> Self {
        Self {
            identity: next_identity(),
        }
    }

    /// Returns the stable output-slot identity.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Creates a normal-priority output-cache request from caller-managed content evidence.
    pub fn cache_request<'a>(
        &self,
        content: impl IntoIterator<Item = &'a ContentChangeHandle>,
    ) -> CacheRequest {
        self.cache_request_with(CacheResidencyPriority::Normal, content)
    }

    /// Creates an output-cache request with an explicit residency priority.
    pub fn cache_request_with<'a>(
        &self,
        priority: CacheResidencyPriority,
        content: impl IntoIterator<Item = &'a ContentChangeHandle>,
    ) -> CacheRequest {
        let (observed_content, revisions): (Vec<_>, Vec<_>) = content
            .into_iter()
            .map(|content| (content.clone(), content.revision()))
            .unzip();
        let stamp = content_stamp_from_revisions(revisions);
        let observed_content = normalize_content_handles(observed_content);

        CacheRequest {
            identity: self.identity,
            stamp,
            observed_content,
            priority,
        }
    }

    /// Creates a normal-priority identity-only keep-alive request.
    pub fn cache_keep_alive(&self) -> CacheKeepAlive {
        self.cache_keep_alive_with(CacheResidencyPriority::Normal)
    }

    /// Creates an identity-only keep-alive request with an explicit residency priority.
    pub fn cache_keep_alive_with(&self, priority: CacheResidencyPriority) -> CacheKeepAlive {
        CacheKeepAlive {
            identity: self.identity,
            priority,
        }
    }
}

impl Default for SurfaceHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SurfaceHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SurfaceHandle")
            .field("identity", &self.identity)
            .finish()
    }
}

impl PartialEq for SurfaceHandle {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Eq for SurfaceHandle {}

impl std::hash::Hash for SurfaceHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

/// Caller-managed evidence that one source of captured content may have changed.
///
/// Clones share a stable identity and one atomic monotonic generation. Pass a clone to every
/// isolated layer which observes this input, keep another clone with the application state or
/// custom widget that detects changes, and call [`mark_changed`](Self::mark_changed) before the
/// next cache lookup whenever the corresponding pixels may differ.
#[derive(Clone)]
pub struct ContentChangeHandle {
    identity: u64,
    generation: Arc<AtomicU64>,
}

impl ContentChangeHandle {
    /// Creates a new content-input identity at generation zero.
    pub fn new() -> Self {
        Self {
            identity: next_identity(),
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns the stable content-input identity.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Returns the currently published content generation.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Marks the observed content as potentially changed and returns its new generation.
    pub fn mark_changed(&self) -> u64 {
        let previous = self
            .generation
            .try_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .expect(
                "content-change generation space exhausted; generations are never reused because reuse could accept obsolete pixels",
            );

        previous + 1
    }

    /// Returns the current content-input identity and generation.
    pub fn revision(&self) -> Revision {
        Revision {
            identity: self.identity,
            generation: self.generation(),
        }
    }
}

impl Default for ContentChangeHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ContentChangeHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContentChangeHandle")
            .field("identity", &self.identity)
            .field("generation", &self.generation())
            .finish_non_exhaustive()
    }
}

impl PartialEq for ContentChangeHandle {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Eq for ContentChangeHandle {}

impl std::hash::Hash for ContentChangeHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

fn next_identity() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    NEXT_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .expect(
            "isolated-layer identity space exhausted; identities are never reused because reuse could alias retained state",
        )
}

fn normalize_content_handles(mut handles: Vec<ContentChangeHandle>) -> Vec<ContentChangeHandle> {
    handles.sort_unstable_by_key(ContentChangeHandle::identity);
    handles.dedup_by_key(|handle| handle.identity());
    handles
}

pub(crate) fn content_stamp<'a>(
    content: impl IntoIterator<Item = &'a ContentChangeHandle>,
) -> ContentStamp {
    let revisions = content
        .into_iter()
        .map(ContentChangeHandle::revision)
        .collect();

    content_stamp_from_revisions(revisions)
}

fn content_stamp_from_revisions(revisions: Vec<Revision>) -> ContentStamp {
    let (revisions, conflicted) = normalize_revisions(revisions);

    ContentStamp {
        revisions,
        conflicted,
    }
}

fn normalize_revisions(mut revisions: Vec<Revision>) -> (Vec<Revision>, bool) {
    revisions.sort_unstable_by_key(|revision| (revision.identity, revision.generation));

    let mut normalized: Vec<Revision> = Vec::with_capacity(revisions.len());
    let mut conflicted = false;

    for revision in revisions {
        let Some(previous) = normalized.last_mut() else {
            normalized.push(revision);
            continue;
        };

        if previous.identity != revision.identity {
            normalized.push(revision);
            continue;
        }

        if previous.generation != revision.generation {
            conflicted = true;
            previous.generation = previous.generation.max(revision.generation);
        }
    }

    (normalized, conflicted)
}

fn canonical(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value.is_nan() {
        return minimum;
    }

    value.clamp(minimum, maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_builders_canonicalize_and_preserve_positioning() {
        let composite = Composite::additive(2.0)
            .with_positioning(CompositePositioning::Subpixel)
            .with_opacity(f32::NAN)
            .with_blend_mode(BlendMode::SourceOver);

        assert_eq!(composite.opacity(), 0.0);
        assert_eq!(composite.blend_mode(), BlendMode::SourceOver);
        assert_eq!(composite.positioning(), CompositePositioning::Subpixel);
        assert_eq!(
            Composite::default().positioning(),
            CompositePositioning::Snapped
        );
    }

    #[test]
    fn surface_clones_share_only_the_output_slot_identity() {
        let surface = SurfaceHandle::new();
        let clone = surface.clone();

        assert_eq!(surface, clone);
        assert_eq!(surface.identity(), clone.identity());
        assert_ne!(surface.identity(), SurfaceHandle::new().identity());
    }

    #[test]
    fn content_change_generations_are_shared_and_monotonic() {
        let content = ContentChangeHandle::new();
        let clone = content.clone();

        assert_eq!(content, clone);
        assert_eq!(clone.mark_changed(), 1);
        assert_eq!(content.generation(), 1);
    }

    #[test]
    fn generation_exhaustion_fails_without_wrapping() {
        let content = ContentChangeHandle {
            identity: 1,
            generation: Arc::new(AtomicU64::new(u64::MAX - 1)),
        };

        assert_eq!(content.mark_changed(), u64::MAX);
        assert!(std::panic::catch_unwind(|| content.mark_changed()).is_err());
        assert_eq!(content.generation(), u64::MAX);
    }

    #[test]
    fn cache_request_normalizes_content_revisions_and_priority() {
        let surface = SurfaceHandle::new();
        let first = ContentChangeHandle::new();
        let second = ContentChangeHandle::new();
        let _ = first.mark_changed();
        let cache_request = surface.cache_request_with(
            CacheResidencyPriority::Protected,
            [&second, &first, &second],
        );

        assert_eq!(cache_request.identity(), surface.identity());
        assert_eq!(cache_request.priority(), CacheResidencyPriority::Protected);
        assert_eq!(
            cache_request.revisions(),
            &[first.revision(), second.revision()]
        );
        assert!(cache_request.has_content_evidence());
        assert!(!cache_request.is_conflicted());
        assert!(cache_request.is_current());
    }

    #[test]
    fn output_keep_alive_contains_only_slot_identity_and_priority() {
        let surface = SurfaceHandle::new();
        let content = ContentChangeHandle::new();
        let request = surface.cache_request([&content]);
        let keep_alive = surface.cache_keep_alive_with(CacheResidencyPriority::Protected);

        assert_eq!(request.identity(), keep_alive.identity());
        assert_eq!(keep_alive.priority(), CacheResidencyPriority::Protected);
    }

    #[test]
    fn empty_content_evidence_fails_closed() {
        let surface = SurfaceHandle::new();
        let request = surface.cache_request([]);

        assert!(!request.has_content_evidence());
        assert!(request.stamp().is_empty());
        assert!(!request.is_current());
    }

    #[test]
    fn cache_request_can_revalidate_and_resnapshot_live_content() {
        let surface = SurfaceHandle::new();
        let content = ContentChangeHandle::new();
        let request = surface.cache_request([&content]);

        assert!(request.is_current());
        assert_eq!(content.mark_changed(), 1);
        assert!(!request.is_current());

        let refreshed = request.resnapshot();
        assert_eq!(refreshed.identity(), request.identity());
        assert_eq!(refreshed.priority(), request.priority());
        assert_eq!(refreshed.revisions(), &[content.revision()]);
        assert!(refreshed.is_current());
        assert_ne!(refreshed.stamp(), request.stamp());
    }

    #[test]
    fn conflicting_duplicate_observations_fail_closed_with_newest_evidence() {
        struct InvalidatingDuplicates<'a> {
            content: &'a ContentChangeHandle,
            remaining: u8,
        }

        impl<'a> Iterator for InvalidatingDuplicates<'a> {
            type Item = &'a ContentChangeHandle;

            fn next(&mut self) -> Option<Self::Item> {
                match self.remaining {
                    2 => {
                        self.remaining = 1;
                        Some(self.content)
                    }
                    1 => {
                        self.remaining = 0;
                        let _ = self.content.mark_changed();
                        Some(self.content)
                    }
                    _ => None,
                }
            }
        }

        let surface = SurfaceHandle::new();
        let content = ContentChangeHandle::new();
        let request = surface.cache_request(InvalidatingDuplicates {
            content: &content,
            remaining: 2,
        });
        let stamp = request.stamp();

        assert!(stamp.is_conflicted());
        assert_eq!(
            stamp.revisions(),
            &[Revision {
                identity: content.identity(),
                generation: 1,
            }]
        );
        assert!(!request.is_current());
    }

    #[test]
    fn core_layer_carries_an_output_request() {
        let surface = SurfaceHandle::new();
        let content = ContentChangeHandle::new();
        let bounds = Rectangle::with_size(crate::Size::new(20.0, 10.0));
        let layer = Layer::new(bounds, bounds).cache_output_with(
            &surface,
            CacheResidencyPriority::Protected,
            [&content],
        );
        let request = layer.output_cache_request.expect("output cache request");

        assert_eq!(request.identity(), surface.identity());
        assert_eq!(request.revisions(), &[content.revision()]);
        assert_eq!(request.priority(), CacheResidencyPriority::Protected);
    }
}
