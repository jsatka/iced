//! Bounded layer-isolated drawing.

mod composite;
mod context;
pub mod effect;
mod pool;
mod recording;
mod retained;
mod scratch;

pub(crate) use scratch::release as release_scratch;
pub use scratch::{ScratchAllocator, ScratchError, ScratchTexture};

pub(crate) use composite::{
    Prepared as PreparedComposite, Storage as CompositeStorage, render as render_composite,
    render_backdrop,
};
pub(crate) use context::{CaptureGrid, Context, Placement};
pub use effect::{
    Context as EffectContext, Effect, EffectStack, Layer, LayerEffect, LayerInputEvidence,
    LayerInputRecords, Pass, Pipeline, PipelineRegistry, Plan, Renderer, Requirements,
    TextureViews,
};
pub(crate) use effect::{Storage as LayerEffectStorage, context as effect_context};
pub(crate) use pool::{Pool, Target};
pub(crate) use recording::{Leaf, Node, PreparedLayer, Recorder, Sequence};
pub(crate) use retained::{
    LeaseTicket, OutputKey, OutputMiss, Registry, StoreDisposition, StoreOutcome,
};

use crate::core::isolated_layer::{CacheKeepAlive, CacheRequest, CacheResidencyPriority};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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
    pub scratch: Vec<Target>,
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
    let backdrop = effects.requirements().needs_backdrop().then_some(1);
    let mut next_output = 1 + usize::from(backdrop.is_some());
    let mut current_output = 0;
    let mut passes = Vec::new();

    for (effect, stage) in effects.iter().enumerate() {
        let stage_input = current_output;

        for pass in 0..stage.passes_len() {
            let requirements = stage.pass_requirements(pass);
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
    /// Baseline bytes owned by the shared free texture pool and retained registry.
    ///
    /// Free textures remain reusable indefinitely while ownership fits. At frame
    /// end, excess ownership is trimmed: oversized free backings first, then the
    /// oldest free backings, normal cached outputs, and finally protected outputs.
    /// A free backing is oversized when it alone exceeds this entire budget.
    /// Active leases and private allocations are not capped by this budget.
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

    /// Creates renderer-local isolated layer limits.
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

/// Isolated layer statistics for the most recently completed renderer draw.
///
/// Counts include processed layers, excluding culled layers, descendants skipped
/// by a cached ancestor, and the synthetic root used for backdrop rendering.
/// Byte counts estimate texture backing storage, including rounded extents; they
/// do not measure driver overhead or total GPU memory usage.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Diagnostics {
    /// Layers whose cached output was reused without rendering their contents.
    pub output_cache_hits: usize,
    /// Cache-eligible layers rendered because no matching output was available.
    pub output_cache_misses: usize,
    /// Layers rendered without caching, including requests that bypass caching.
    pub uncached_renders: usize,
    /// Newly allocated texture bytes during this draw, including layer, effect,
    /// scratch, and root intermediates, even if discarded before the draw ends.
    /// Reusing a pooled or cached texture contributes zero.
    pub allocated_bytes: u64,
    /// Free pooled texture bytes after draw cleanup and budget enforcement.
    pub pool_bytes: u64,
    /// Normal-priority cached texture bytes after cleanup and budget enforcement.
    pub normal_priority_bytes: u64,
    /// Protected-priority cached texture bytes after cleanup and budget enforcement.
    /// Protection affects eviction order; these textures are not pinned.
    pub protected_priority_bytes: u64,
}

impl Diagnostics {
    /// Returns the number of processed isolated layers in this draw.
    pub fn layer_count(&self) -> usize {
        self.output_cache_hits + self.output_cache_misses + self.uncached_renders
    }

    /// Returns the disjoint sum of pooled and cached texture bytes after this draw.
    pub fn retained_bytes(&self) -> u64 {
        self.pool_bytes + self.normal_priority_bytes + self.protected_priority_bytes
    }

    pub(crate) fn record_layer(&mut self, valid: bool, cacheable: bool) {
        if valid {
            self.output_cache_hits += 1;
        } else if cacheable {
            self.output_cache_misses += 1;
        } else {
            self.uncached_renders += 1;
        }
    }

    pub(crate) fn record_allocation(&mut self, bytes: u64, reused: bool) {
        if !reused {
            self.allocated_bytes += bytes;
        }
    }
}

#[derive(Debug, Clone)]
struct PendingKeepAlive {
    request: CacheKeepAlive,
}

impl PendingKeepAlive {
    fn new(request: CacheKeepAlive) -> Self {
        Self { request }
    }

    fn merge(&mut self, incoming: CacheKeepAlive) {
        if self.request.priority() != incoming.priority()
            && incoming.priority() == CacheResidencyPriority::Normal
        {
            self.request = incoming;
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
    missing_content: HashSet<u64>,
    previous_missing_content: HashSet<u64>,
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
            missing_content: HashSet::new(),
            previous_missing_content: HashSet::new(),
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

    pub(crate) fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.diagnostics = Diagnostics::default();
        std::mem::swap(
            &mut self.missing_content,
            &mut self.previous_missing_content,
        );
        self.missing_content.clear();
        let _ = self.registry.recover_abandoned(self.frame);

        for pending in std::mem::take(&mut self.frame_keep_alives).into_values() {
            let _ = self.registry.keep_alive(&pending.request, self.frame);
        }
    }

    pub(crate) fn finish_frame(&mut self) {
        let audit = self.registry.finish_frame(self.frame);
        let swept = self.registry.sweep(self.frame, self.limits.grace_frames);
        self.release_targets(swept.released);

        self.enforce_budget();
        self.refresh_usage();

        debug_assert!(
            self.diagnostics.retained_bytes() <= self.limits.budget_bytes
                || (audit.output_leases > 0 && self.pool.bytes() == 0),
            "GPU target ownership exceeds the renderer-local budget: {} > {}",
            self.diagnostics.retained_bytes(),
            self.limits.budget_bytes,
        );
    }

    pub(crate) fn release_targets(&mut self, targets: Vec<Target>) {
        for target in targets {
            self.pool.release(target);
        }
    }

    pub(crate) fn record_output_miss(&mut self, identity: u64, miss: OutputMiss) {
        if self.should_report_missing_content(identity, miss) {
            log::error!(
                "isolated layer surface {identity} requested output caching without content evidence on draw {}; supply content handles covering the layer's inputs",
                self.frame,
            );
        }
    }

    fn should_report_missing_content(&mut self, identity: u64, miss: OutputMiss) -> bool {
        miss == OutputMiss::MissingContentEvidence
            && self.missing_content.insert(identity)
            && !self.previous_missing_content.contains(&identity)
    }

    pub(crate) fn record_store(&mut self, outcome: StoreOutcome) -> StoreDisposition {
        self.release_targets(outcome.released);
        outcome.disposition
    }

    fn enforce_budget(&mut self) {
        let registry_bytes = self.registry.bytes();
        let maximum = self.limits.budget_bytes.saturating_sub(registry_bytes);
        let _ = self.pool.trim_to_bytes(maximum, self.limits.budget_bytes);

        let pool_bytes = self.pool.bytes();
        if registry_bytes <= self.limits.budget_bytes.saturating_sub(pool_bytes) {
            return;
        }
        let outcome = self.registry.evict_to_bytes(
            self.limits.budget_bytes.saturating_sub(pool_bytes),
            self.frame,
        );

        for evicted in outcome.evicted {
            drop(evicted.target);
        }

        debug_assert_eq!(outcome.remaining_bytes, self.registry.bytes());
    }

    fn refresh_usage(&mut self) {
        let resident = self.registry.resident_bytes();
        self.diagnostics.pool_bytes = self.pool.bytes();
        self.diagnostics.normal_priority_bytes = resident.normal;
        self.diagnostics.protected_priority_bytes = resident.protected;
        debug_assert_eq!(resident.total(), self.registry.bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::isolated_layer::SurfaceHandle;

    #[derive(Debug, Clone, PartialEq)]
    struct PlannedStage {
        passes: usize,
        requirements: effect::Requirements,
    }

    impl PlannedStage {
        fn new(passes: usize, requirements: effect::Requirements) -> Self {
            Self {
                passes,
                requirements,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct PlannedPass {
        pass: usize,
        requirements: effect::Requirements,
    }

    impl effect::LayerEffect for PlannedStage {
        fn plan(&self, plan: &mut effect::Plan<'_, Self>) {
            for pass in 0..self.passes {
                plan.push(PlannedPass {
                    pass,
                    requirements: self.requirements,
                });
            }
        }
    }

    impl effect::Pass<PlannedStage> for PlannedPass {
        type Prepared = ();

        fn requirements(&self, _effect: &PlannedStage) -> effect::Requirements {
            self.requirements
        }

        fn prepare(
            &self,
            _effect: &PlannedStage,
            _pipelines: &mut effect::PipelineRegistry<'_>,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _scratch: &mut effect::ScratchAllocator<'_>,
            _context: &effect::Context,
            _views: effect::TextureViews<'_>,
        ) {
        }

        fn encode(
            &self,
            _effect: &PlannedStage,
            _pipelines: &effect::PipelineRegistry<'_>,
            _prepared: &Self::Prepared,
            _encoder: &mut wgpu::CommandEncoder,
            _context: &effect::Context,
            _views: effect::TextureViews<'_>,
        ) {
        }
    }

    #[test]
    fn production_pass_plan_chains_stage_outputs_and_scopes_backdrop_per_stage() {
        let effects = EffectStack::new()
            .with(PlannedStage::new(2, effect::Requirements::new()))
            .with(PlannedStage::new(
                1,
                effect::Requirements::new()
                    .with_backdrop()
                    .writes_every_pixel(),
            ))
            .with(PlannedStage::new(
                3,
                effect::Requirements::new().writes_every_pixel(),
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
            .with(PlannedStage::new(32, effect::Requirements::new()))
            .with(PlannedStage::new(17, effect::Requirements::new()));

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
    fn empty_stages_forward_without_targets_or_backdrop() {
        let effects = EffectStack::new()
            .with(PlannedStage::new(
                0,
                effect::Requirements::new().with_backdrop(),
            ))
            .with(PlannedStage::new(1, effect::Requirements::new()))
            .with(PlannedStage::new(
                0,
                effect::Requirements::new().with_backdrop(),
            ));

        let plan = plan_effect_passes(&effects);

        assert_eq!(plan.backdrop, None);
        assert_eq!(plan.target_count, 2);
        assert_eq!(plan.output, 1);
        assert_eq!(
            plan.passes,
            vec![PlannedEffectPass {
                effect: 1,
                pass: 0,
                stage_input: 0,
                previous: 0,
                output: 1,
                uses_backdrop: false,
                writes_every_pixel: false,
            }]
        );
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

        state.begin_frame();
        assert!(state.frame_keep_alives.is_empty());
        state.diagnostics.record_layer(false, true);

        state.begin_frame();
        assert_eq!(state.diagnostics, Diagnostics::default());
    }

    #[test]
    fn diagnostics_separate_outcomes_and_allocation_from_residency() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.record_layer(true, true);
        diagnostics.record_layer(false, true);
        diagnostics.record_layer(false, false);
        assert_eq!(diagnostics.layer_count(), 3);
        assert_eq!(diagnostics.output_cache_hits, 1);
        assert_eq!(diagnostics.output_cache_misses, 1);
        assert_eq!(diagnostics.uncached_renders, 1);
        diagnostics.record_allocation(256, false);
        diagnostics.record_allocation(512, true);
        assert_eq!(diagnostics.allocated_bytes, 256);
        assert_eq!(diagnostics.retained_bytes(), 0);
    }
    #[test]
    fn missing_content_errors_are_deduplicated_across_consecutive_draws() {
        let mut state = State::default();
        let missing = OutputMiss::MissingContentEvidence;
        state.begin_frame();
        assert!(!state.should_report_missing_content(1, OutputMiss::ContentChanged));
        assert!(state.should_report_missing_content(1, missing));
        assert!(!state.should_report_missing_content(1, missing));
        state.begin_frame();
        assert!(!state.should_report_missing_content(1, missing));
        assert!(state.should_report_missing_content(2, missing));
        state.begin_frame();
        state.begin_frame();
        assert!(state.should_report_missing_content(1, missing));
    }
}
