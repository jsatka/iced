//! Declarative effect contracts for GPU texture-backed isolated layers.

use crate::core::{Padding, Rectangle, Size, isolated_layer};
use crate::graphics::futures::{MaybeSend, MaybeSync};

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::ops::Index;
use std::sync::Arc;

/// Resource requirements of one isolated-layer effect pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Requirements {
    backdrop: bool,
    writes_every_pixel: bool,
}

impl Requirements {
    /// Creates requirements with no backdrop or output-initialization guarantee.
    pub const fn new() -> Self {
        Self {
            backdrop: false,
            writes_every_pixel: false,
        }
    }

    /// Declares a parent-prefix backdrop input.
    pub const fn with_backdrop(mut self) -> Self {
        self.backdrop = true;
        self
    }

    /// Declares that the pass initializes every output pixel.
    ///
    /// The guarantee must hold for every pipeline or target-dependent algorithm
    /// this descriptor may select. It lets the renderer skip its default
    /// transparent clear; an explicit clear encoded by the pass also satisfies
    /// the guarantee.
    pub const fn writes_every_pixel(mut self) -> Self {
        self.writes_every_pixel = true;
        self
    }

    /// Returns whether a parent-prefix snapshot is required.
    pub const fn needs_backdrop(self) -> bool {
        self.backdrop
    }

    /// Returns whether every output is fully initialized by its pass.
    pub const fn fully_overwrites(self) -> bool {
        self.writes_every_pixel
    }
}

/// Target facts available while preparing and encoding an isolated layer pass.
#[derive(Debug, Clone, Copy)]
pub struct Context {
    /// Bounds represented by the valid physical pixels in each supplied target.
    ///
    /// They include aggregate effect expansion and use the same coordinate
    /// system as [`Layer::bounds`].
    pub represented_bounds: Rectangle,
    /// Bounds of the unexpanded widget content.
    ///
    /// They use the same coordinate system as [`Self::represented_bounds`].
    pub content_bounds: Rectangle,
    /// Valid physical pixels in each supplied target.
    pub physical_size: Size<u32>,
    /// Full pooled texture extent.
    pub backing_extent: Size<u32>,
    /// Valid normalized texture-coordinate maximum.
    pub valid_uv: [f32; 2],
    /// Scale applied when rasterizing layer bounds into physical target pixels.
    pub scale_factor: f32,
    /// Target format.
    pub format: wgpu::TextureFormat,
}

/// Texture views supplied to one isolated layer pass.
pub struct TextureViews<'a> {
    /// The stable input to this effect stage.
    ///
    /// This is the captured child for the first stage and the completed output
    /// of the preceding stage for every later stage.
    pub stage_input: &'a wgpu::TextureView,
    /// The declared parent-prefix snapshot, when requested.
    pub backdrop: Option<&'a wgpu::TextureView>,
    /// The output of the preceding local pass, or `stage_input` for the first pass.
    pub previous: &'a wgpu::TextureView,
    /// The dedicated output of this pass.
    pub output: &'a wgpu::TextureView,
}

/// Plain data describing one stage in an isolated-layer effect stack.
///
/// [`Effect::new`] invokes [`LayerEffect::plan`] exactly once and freezes the
/// resulting pass sequence, each pass's requirements, the expansion, and
/// translation behavior. Construct a replacement [`Effect`] to change any of
/// those values. Mutable pixel inputs may still be described by
/// [`LayerEffect::record_inputs`] using stable identity plus a monotonic revision,
/// or may opt out of output caching by marking their inputs volatile.
///
/// The count-based API was replaced directly. See the
/// [isolated-layer effect migration guide](https://github.com/iced-rs/iced/blob/master/docs/isolated-layer-effect-migration.md)
/// for a complete custom-effect example and the lifetime rules.
pub trait LayerEffect: Debug + Clone + PartialEq + MaybeSend + MaybeSync + 'static {
    /// Builds the ordered executable pass plan for this effect.
    ///
    /// The plan is CPU-only and construction-time immutable. Adding a pass is
    /// authoritative: the renderer derives allocation, preparation, encoding,
    /// cache evidence, and final-output selection from the registered entries.
    fn plan(&self, plan: &mut Plan<'_, Self>)
    where
        Self: Sized;

    /// Returns the capture expansion around the widget bounds.
    ///
    /// Every side must be finite and non-negative. An [`EffectStack`] rejects
    /// an invalid expansion instead of constructing or allocating an
    /// undersized target.
    fn expansion(&self) -> Padding {
        Padding::ZERO
    }

    /// Records a snapshot of effect inputs that may change the rasterized
    /// pixel output. Recorded value [`PartialEq`] will be used by renderer
    /// to determine whether a previous cached output may be reused.
    ///
    /// Effects with externally mutable inputs must override this
    /// method to record stable resource revisions or call
    /// [`LayerInputRecords::mark_volatile`] to prevent using past cached
    /// outputs.
    fn record_inputs(&self, inputs: &mut LayerInputRecords) {
        inputs.record(self);
    }

    /// Returns whether translating the capture can leave this effect's output
    /// unchanged.
    ///
    /// The default is conservative because an effect can inspect the absolute
    /// [`Context::represented_bounds`] and [`Context::content_bounds`].
    fn is_translation_invariant(&self) -> bool {
        false
    }
}

/// One logical texture-to-texture operation in an isolated-layer effect plan.
///
/// A pass receives one dedicated full-size renderer output. Its prepared state
/// may own supplementary GPU resources, and [`Pass::encode`] may encode multiple
/// render or compute operations through the raw command encoder. The pass must
/// leave its supplied output fully valid when encoding returns.
///
/// Supplementary resources are private to the pass: the renderer does not pool,
/// budget, initialize, validate dependencies for, or report their internal GPU
/// operations. Put per-application ownership in [`Pass::Prepared`], initialize
/// every sampled region, and write the final result into [`TextureViews::output`].
///
/// A descriptor should contain owned, immutable, comparable configuration only.
/// Its construction-time clone becomes cache evidence. If it refers to mutable
/// external pixel inputs, the enclosing effect must record their revisions or
/// mark itself volatile through [`LayerEffect::record_inputs`].
pub trait Pass<E: LayerEffect>:
    Debug + Clone + PartialEq + MaybeSend + MaybeSync + 'static
{
    /// Renderer-local resources prepared for this pass.
    ///
    /// The value is retained through encoding and dropped before the renderer
    /// returns this application's transient targets to its pool. It may own
    /// textures, views, bind groups, buffers, and an internal operation schedule.
    type Prepared: Any + MaybeSend + MaybeSync;

    /// Returns this pass's frozen backdrop and output-initialization requirements.
    ///
    /// This method runs when the descriptor is added to the construction-time
    /// plan. Its answer must cover every compatible choice made later from
    /// [`Context`] or device capabilities.
    fn requirements(&self, _effect: &E) -> Requirements {
        Requirements::new()
    }

    /// Prepares renderer-local resources for this pass.
    fn prepare(
        &self,
        effect: &E,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        context: &Context,
        views: TextureViews<'_>,
    ) -> Self::Prepared;

    /// Encodes this pass.
    ///
    /// `views.output` is dedicated to this pass and never aliases an input.
    fn encode(
        &self,
        effect: &E,
        pipelines: &PipelineRegistry<'_>,
        prepared: &Self::Prepared,
        encoder: &mut wgpu::CommandEncoder,
        context: &Context,
        views: TextureViews<'_>,
    );
}

/// Construction-only builder for an effect's executable pass sequence.
pub struct Plan<'a, E: LayerEffect> {
    effect: &'a E,
    passes: Vec<PassEntry<E>>,
}

impl<E: LayerEffect> Plan<'_, E> {
    /// Appends one owned pass descriptor to the effect plan.
    ///
    /// Different calls may use different concrete pass and prepared-resource
    /// types. Descriptor values are included automatically in output-cache
    /// evidence once at construction; external mutable resources still belong in
    /// [`LayerEffect::record_inputs`].
    pub fn push<P>(&mut self, pass: P)
    where
        P: Pass<E>,
    {
        let requirements = pass.requirements(self.effect);
        self.passes.push(PassEntry {
            requirements,
            evidence: InputAtom(Arc::new(pass.clone())),
            pass: Box::new(pass),
        });
    }
}

impl<E: LayerEffect> Debug for Plan<'_, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Plan")
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

fn build_plan<E: LayerEffect>(effect: &E) -> Vec<PassEntry<E>> {
    let mut plan = Plan {
        effect,
        passes: Vec::new(),
    };
    effect.plan(&mut plan);
    plan.passes
}

/// Records retained, directly comparable snapshots of effect inputs.
#[derive(Debug, Default)]
pub struct LayerInputRecords {
    atoms: Vec<InputAtom>,
    volatile: bool,
}

impl LayerInputRecords {
    /// Creates an empty input record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an owned clone of a borrowed value.
    pub fn record<T>(&mut self, value: &T)
    where
        T: Any + Clone + Debug + PartialEq + MaybeSend + MaybeSync,
    {
        self.atoms.push(InputAtom(Arc::new(value.clone())));
    }

    /// Records an explicit stable resource identity and monotonic generation.
    pub fn record_revision(&mut self, identity: u64, generation: u64) {
        self.record(&ResourceRevision {
            identity,
            generation,
        });
    }

    /// Records the current revision of a shared content-change handle.
    pub fn depend_on(&mut self, dependency: &isolated_layer::ContentChangeHandle) {
        self.record_revision(dependency.identity(), dependency.generation());
    }

    /// Declares that this effect cannot provide stable, complete input evidence.
    ///
    /// One volatile stage makes output-cache lookup and storage ineligible for
    /// the complete stack.
    pub fn mark_volatile(&mut self) {
        self.volatile = true;
    }

    /// Finishes input recoding.
    pub fn finish(self) -> LayerInputEvidence {
        LayerInputEvidence::new(self.atoms, self.volatile)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResourceRevision {
    identity: u64,
    generation: u64,
}

/// An exact, cloneable snapshot of an effect stack's inputs and structure.
#[derive(Clone)]
pub struct LayerInputEvidence {
    atoms: Arc<[InputAtom]>,
    volatile: bool,
}

impl LayerInputEvidence {
    fn new(atoms: Vec<InputAtom>, volatile: bool) -> Self {
        Self {
            atoms: atoms.into(),
            volatile,
        }
    }

    /// Returns whether any stage explicitly declined retained-output reuse.
    pub fn is_volatile(&self) -> bool {
        self.volatile
    }

    /// Returns whether both snapshots are stable and exactly equal.
    ///
    /// Volatile evidence deliberately never matches, including itself.
    pub fn matches(&self, other: &Self) -> bool {
        !self.volatile && !other.volatile && self == other
    }
}

impl Debug for LayerInputEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LayerInputEvidence")
            .field("atoms", &self.atoms)
            .field("volatile", &self.volatile)
            .finish()
    }
}

impl PartialEq for LayerInputEvidence {
    fn eq(&self, other: &Self) -> bool {
        self.volatile == other.volatile && self.atoms.as_ref() == other.atoms.as_ref()
    }
}

#[derive(Clone)]
struct InputAtom(Arc<dyn ExactInput>);

impl Debug for InputAtom {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl PartialEq for InputAtom {
    fn eq(&self, other: &Self) -> bool {
        self.0.equals(other.0.as_ref())
    }
}

trait ExactInput: MaybeSend + MaybeSync {
    fn as_any(&self) -> &dyn Any;
    fn equals(&self, other: &dyn ExactInput) -> bool;
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T> ExactInput for T
where
    T: Any + Debug + PartialEq + MaybeSend + MaybeSync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn equals(&self, other: &dyn ExactInput) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }

    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, formatter)
    }
}

/// A cloneable, type-erased isolated-layer effect stage.
#[derive(Clone)]
pub struct Effect(Arc<dyn Stored>);

impl Effect {
    /// Builds and erases one concrete effect while retaining it for recollection.
    pub fn new(effect: impl LayerEffect) -> Self {
        let expansion = effect.expansion();
        let translation_invariant = effect.is_translation_invariant();
        let passes = build_plan(&effect);
        let requirements = summarize_requirements(passes.iter().map(|entry| entry.requirements));

        Self(Arc::new(BlackBox {
            effect,
            passes,
            requirements,
            expansion,
            translation_invariant,
        }))
    }

    /// Returns the requirements derived from this effect's stored passes.
    pub fn requirements(&self) -> Requirements {
        self.0.requirements()
    }

    /// Returns the effect's construction-time capture expansion.
    pub fn expansion(&self) -> Padding {
        self.0.expansion()
    }

    /// Contributes the effect-owned portion of its input evidence.
    pub fn contribute_inputs(&self, inputs: &mut LayerInputRecords) {
        self.0.contribute_inputs(inputs);
    }

    /// Returns the effect's construction-time translation behavior.
    pub fn is_translation_invariant(&self) -> bool {
        self.0.is_translation_invariant()
    }

    pub(crate) fn passes_len(&self) -> usize {
        self.0.passes_len()
    }

    pub(crate) fn pass_requirements(&self, pass: usize) -> Requirements {
        self.0.pass_requirements(pass)
    }

    pub(crate) fn stored(&self) -> &dyn Stored {
        self.0.as_ref()
    }
}

impl Debug for Effect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.0.as_ref(), formatter)
    }
}

/// An ordered heterogeneous sequence of isolated-layer effect stages.
#[derive(Debug, Clone, Default)]
pub struct EffectStack {
    effects: Vec<Effect>,
}

impl EffectStack {
    /// Creates an empty stack, which represents an effect-free isolated layer.
    pub const fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Creates a stack containing one effect stage.
    pub fn from_effect(effect: impl LayerEffect) -> Self {
        Self {
            effects: vec![Effect::new(effect)],
        }
    }

    /// Appends one concrete effect stage.
    pub fn push(&mut self, effect: impl LayerEffect) {
        self.effects.push(Effect::new(effect));
    }

    /// Appends an already-erased effect stage.
    pub fn push_erased(&mut self, effect: Effect) {
        self.effects.push(effect);
    }

    /// Appends one effect stage and returns the updated stack.
    pub fn with(mut self, effect: impl LayerEffect) -> Self {
        self.push(effect);
        self
    }

    /// Returns the number of effect stages.
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Returns whether the stack has no effect stages.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Returns one effect stage by index.
    pub fn get(&self, index: usize) -> Option<&Effect> {
        self.effects.get(index)
    }

    /// Iterates the effect stages in execution order.
    pub fn iter(&self) -> std::slice::Iter<'_, Effect> {
        self.effects.iter()
    }

    /// Iterates per-stage requirements in execution order.
    ///
    /// Renderers must retain these boundaries when deciding whether each pass
    /// output needs clearing; the aggregate summary is not sufficient for that
    /// decision.
    pub fn stage_requirements(
        &self,
    ) -> impl ExactSizeIterator<Item = Requirements> + DoubleEndedIterator + '_ {
        self.effects.iter().map(Effect::requirements)
    }

    /// Returns a conservative aggregate requirement summary.
    ///
    /// Backdrop use is the union of every stored pass. Full overwrite is true
    /// only when at least one pass exists and every pass guarantees it.
    pub fn requirements(&self) -> Requirements {
        summarize_requirements(
            self.effects.iter().flat_map(|effect| {
                (0..effect.passes_len()).map(|pass| effect.pass_requirements(pass))
            }),
        )
    }

    /// Returns the canonical component-wise sum of every stage's expansion.
    ///
    /// Signed zero is normalized to positive zero. A non-finite or negative
    /// side, or a component sum that cannot be represented by a finite `f32`,
    /// makes the aggregate invalid. Sums are accumulated as `f64` so a series
    /// of otherwise valid stage expansions cannot overflow before validation.
    pub fn expansion(&self) -> Option<Padding> {
        let mut total = [0.0f64; 4];

        for effect in &self.effects {
            let expansion = canonical_expansion(effect.expansion())?;

            total[0] += f64::from(expansion.top);
            total[1] += f64::from(expansion.right);
            total[2] += f64::from(expansion.bottom);
            total[3] += f64::from(expansion.left);
        }

        Some(Padding {
            top: checked_expansion_sum(total[0])?,
            right: checked_expansion_sum(total[1])?,
            bottom: checked_expansion_sum(total[2])?,
            left: checked_expansion_sum(total[3])?,
        })
    }

    /// Returns whether every effect stage is translation invariant.
    ///
    /// An empty effect stack is effect-invariant.
    pub fn is_translation_invariant(&self) -> bool {
        self.effects.iter().all(Effect::is_translation_invariant)
    }

    /// Collects exact effect inputs plus framework-owned stack structure.
    pub fn input_evidence(&self) -> LayerInputEvidence {
        let mut inputs = LayerInputRecords::new();
        inputs.record(&FrameworkInput::Stack {
            schema: INPUT_SCHEMA,
            stages: self.effects.len(),
        });

        for (index, effect) in self.effects.iter().enumerate() {
            let expansion = effect.expansion();
            let expansion = canonical_expansion(expansion).unwrap_or(expansion);
            inputs.record(&FrameworkInput::Stage {
                index,
                effect_type: effect.stored().effect_type_id(),
                expansion: expansion_bits(expansion),
                translation_invariant: effect.is_translation_invariant(),
            });
            effect.contribute_inputs(&mut inputs);
            for pass in 0..effect.passes_len() {
                inputs.record(&FrameworkInput::Pass {
                    index: pass,
                    pass_type: effect.stored().pass_type_id(pass),
                    requirements: effect.pass_requirements(pass),
                });
                effect.stored().contribute_pass_inputs(pass, &mut inputs);
                inputs.record(&FrameworkInput::EndPass { index: pass });
            }
            inputs.record(&FrameworkInput::EndStage { index });
        }

        inputs.finish()
    }

    /// Recollects evidence from the same retained erased effect instances.
    pub fn recollect_input_evidence(&self) -> LayerInputEvidence {
        self.input_evidence()
    }

    /// Recollects the stack and compares it with an earlier stable snapshot.
    pub fn inputs_match(&self, recorded: &LayerInputEvidence) -> bool {
        recorded.matches(&self.recollect_input_evidence())
    }
}

fn summarize_requirements(requirements: impl IntoIterator<Item = Requirements>) -> Requirements {
    let mut has_passes = false;
    let mut backdrop = false;
    let mut writes_every_pixel = true;

    for requirements in requirements {
        has_passes = true;
        backdrop |= requirements.needs_backdrop();
        writes_every_pixel &= requirements.fully_overwrites();
    }

    Requirements {
        backdrop,
        writes_every_pixel: has_passes && writes_every_pixel,
    }
}

impl Index<usize> for EffectStack {
    type Output = Effect;

    fn index(&self, index: usize) -> &Self::Output {
        &self.effects[index]
    }
}

impl Extend<Effect> for EffectStack {
    fn extend<T: IntoIterator<Item = Effect>>(&mut self, effects: T) {
        self.effects.extend(effects);
    }
}

impl FromIterator<Effect> for EffectStack {
    fn from_iter<T: IntoIterator<Item = Effect>>(effects: T) -> Self {
        Self {
            effects: effects.into_iter().collect(),
        }
    }
}

impl IntoIterator for EffectStack {
    type Item = Effect;
    type IntoIter = std::vec::IntoIter<Effect>;

    fn into_iter(self) -> Self::IntoIter {
        self.effects.into_iter()
    }
}

impl<'a> IntoIterator for &'a EffectStack {
    type Item = &'a Effect;
    type IntoIter = std::slice::Iter<'a, Effect>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

fn canonical_expansion(expansion: Padding) -> Option<Padding> {
    Some(Padding {
        top: canonical_expansion_side(expansion.top)?,
        right: canonical_expansion_side(expansion.right)?,
        bottom: canonical_expansion_side(expansion.bottom)?,
        left: canonical_expansion_side(expansion.left)?,
    })
}

fn canonical_expansion_side(value: f32) -> Option<f32> {
    if !value.is_finite() || value < 0.0 {
        None
    } else if value == 0.0 {
        Some(0.0)
    } else {
        Some(value)
    }
}

fn checked_expansion_sum(value: f64) -> Option<f32> {
    if !value.is_finite() || value > f64::from(f32::MAX) {
        return None;
    }

    canonical_expansion_side(value as f32)
}

fn expansion_bits(expansion: Padding) -> [u32; 4] {
    [
        expansion.top.to_bits(),
        expansion.right.to_bits(),
        expansion.bottom.to_bits(),
        expansion.left.to_bits(),
    ]
}

const INPUT_SCHEMA: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FrameworkInput {
    Stack {
        schema: u32,
        stages: usize,
    },
    Stage {
        index: usize,
        effect_type: TypeId,
        expansion: [u32; 4],
        translation_invariant: bool,
    },
    Pass {
        index: usize,
        pass_type: TypeId,
        requirements: Requirements,
    },
    EndPass {
        index: usize,
    },
    EndStage {
        index: usize,
    },
}

/// Long-lived compiled state shared by every effect value that requests its concrete type.
pub trait Pipeline: Any + MaybeSend + MaybeSync {
    /// Creates compiled state lazily on first use.
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self
    where
        Self: Sized;
}

enum Access<'a> {
    Read(&'a Storage),
    Write(&'a mut Storage),
}

/// An engine-wide, typed registry of lazily-created effect pipelines.
pub struct PipelineRegistry<'a> {
    access: Access<'a>,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    format: wgpu::TextureFormat,
}

impl PipelineRegistry<'_> {
    /// Returns an initialized pipeline of type `P`, creating it on first use.
    pub fn get_or_init<P: Pipeline>(&mut self) -> &mut P {
        let Access::Write(storage) = &mut self.access else {
            panic!("layer-effect pipelines can only be initialized during preparation");
        };

        storage.get_or_init::<P>(self.device, self.queue, self.format)
    }

    /// Returns a previously initialized pipeline of type `P`.
    pub fn get<P: Pipeline>(&self) -> Option<&P> {
        match &self.access {
            Access::Read(storage) => storage.get::<P>(),
            Access::Write(storage) => storage.get::<P>(),
        }
    }
}

/// A WGPU renderer that can apply an effect stack to isolated ordinary drawing.
pub trait Renderer: crate::core::Renderer {
    /// Starts an isolated-layer effect scope.
    ///
    /// The scope is an exact paint-order barrier: ordinary primitives recorded
    /// afterward paint after the layer output, regardless of their usual
    /// intra-layer batch order.
    fn start_isolated_layer_effects(&mut self, layer: Layer, effects: EffectStack);

    /// Ends an isolated-layer effect scope.
    fn end_isolated_layer_effects(&mut self);

    /// Draws ordinary renderer calls into an isolated-layer effect scope.
    fn with_isolated_layer_effects(
        &mut self,
        layer: Layer,
        effects: EffectStack,
        draw: impl FnOnce(&mut Self),
    ) {
        self.start_isolated_layer_effects(layer, effects);
        draw(self);
        self.end_isolated_layer_effects();
    }
}

/// Bounds, output-cache request, and fixed-function composition of isolated drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Capture and output bounds.
    ///
    /// They use the same coordinate system as ordinary renderer calls recorded
    /// inside the layer scope.
    pub bounds: Rectangle,
    /// Bounds of the unexpanded content.
    ///
    /// They use the same coordinate system as [`Self::bounds`].
    pub content_bounds: Rectangle,
    /// Final composite clip, expressed in the same coordinate system as
    /// [`Self::bounds`].
    ///
    /// This clip does not restrict capture. An effect may sample captured
    /// pixels inside [`bounds`](Self::bounds) but outside this rectangle before
    /// the output is clipped during composition.
    pub clip: Rectangle,
    /// Fixed-function composition settings.
    pub composite: isolated_layer::Composite,
    /// Whether captured child pixels can depend on absolute translation.
    ///
    /// When true, absolute represented and content bounds must participate in
    /// output-cache validity even if every effect is translation invariant.
    pub content_depends_on_translation: bool,
    /// Optional final-output cache identity, content evidence, and priority.
    pub output_cache_request: Option<isolated_layer::CacheRequest>,
}

impl Layer {
    /// Creates a source-over isolated layer request.
    pub fn new(bounds: Rectangle, clip: Rectangle) -> Self {
        Self {
            bounds,
            content_bounds: bounds,
            clip,
            composite: isolated_layer::Composite::default(),
            content_depends_on_translation: false,
            output_cache_request: None,
        }
    }

    /// Sets the unexpanded widget content bounds.
    pub fn content_bounds(mut self, bounds: Rectangle) -> Self {
        self.content_bounds = bounds;
        self
    }

    /// Sets fixed-function composition settings.
    pub fn composite(mut self, composite: isolated_layer::Composite) -> Self {
        self.composite = composite;
        self
    }

    /// Declares whether captured child pixels depend on absolute translation.
    pub fn content_depends_on_translation(mut self, depends: bool) -> Self {
        self.content_depends_on_translation = depends;
        self
    }

    /// Requests caching of the final pre-composite output.
    pub fn cache_output<'a>(
        self,
        surface: &isolated_layer::SurfaceHandle,
        content: impl IntoIterator<Item = &'a isolated_layer::ContentChangeHandle>,
    ) -> Self {
        self.cache_output_with(
            surface,
            isolated_layer::CacheResidencyPriority::Normal,
            content,
        )
    }

    /// Requests output caching with an explicit residency priority.
    pub fn cache_output_with<'a>(
        mut self,
        surface: &isolated_layer::SurfaceHandle,
        priority: isolated_layer::CacheResidencyPriority,
        content: impl IntoIterator<Item = &'a isolated_layer::ContentChangeHandle>,
    ) -> Self {
        self.output_cache_request = Some(surface.cache_request_with(priority, content));
        self
    }

    /// Sets a previously constructed output-cache request.
    pub fn cache_output_request(mut self, request: isolated_layer::CacheRequest) -> Self {
        self.output_cache_request = Some(request);
        self
    }
}

#[derive(Debug)]
struct PassEntry<E: LayerEffect> {
    requirements: Requirements,
    evidence: InputAtom,
    pass: Box<dyn StoredPass<E>>,
}

trait StoredPass<E: LayerEffect>: Debug + MaybeSend + MaybeSync {
    fn pass_type_id(&self) -> TypeId;

    fn prepare(
        &self,
        effect: &E,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        context: &Context,
        views: TextureViews<'_>,
    ) -> Box<dyn Erased>;

    fn encode(
        &self,
        effect: &E,
        pipelines: &PipelineRegistry<'_>,
        prepared: &dyn Erased,
        encoder: &mut wgpu::CommandEncoder,
        context: &Context,
        views: TextureViews<'_>,
    );
}

impl<E, P> StoredPass<E> for P
where
    E: LayerEffect,
    P: Pass<E>,
{
    fn pass_type_id(&self) -> TypeId {
        TypeId::of::<P>()
    }

    fn prepare(
        &self,
        effect: &E,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        context: &Context,
        views: TextureViews<'_>,
    ) -> Box<dyn Erased> {
        Box::new(Pass::prepare(
            self, effect, pipelines, device, queue, context, views,
        ))
    }

    fn encode(
        &self,
        effect: &E,
        pipelines: &PipelineRegistry<'_>,
        prepared: &dyn Erased,
        encoder: &mut wgpu::CommandEncoder,
        context: &Context,
        views: TextureViews<'_>,
    ) {
        let prepared = prepared
            .as_any()
            .downcast_ref::<P::Prepared>()
            .expect("isolated-layer prepared resources must match their pass descriptor");
        Pass::encode(self, effect, pipelines, prepared, encoder, context, views);
    }
}

pub(crate) trait Stored: Debug + MaybeSend + MaybeSync {
    fn effect_type_id(&self) -> TypeId;
    fn requirements(&self) -> Requirements;
    fn expansion(&self) -> Padding;
    fn contribute_inputs(&self, inputs: &mut LayerInputRecords);
    fn is_translation_invariant(&self) -> bool;
    fn passes_len(&self) -> usize;
    fn pass_requirements(&self, pass: usize) -> Requirements;
    fn pass_type_id(&self, pass: usize) -> TypeId;
    fn contribute_pass_inputs(&self, pass: usize, inputs: &mut LayerInputRecords);

    fn prepare(
        &self,
        storage: &mut Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        pass: usize,
        context: &Context,
        views: TextureViews<'_>,
    ) -> Box<dyn Erased>;

    fn encode(
        &self,
        storage: &Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        prepared: &dyn Erased,
        encoder: &mut wgpu::CommandEncoder,
        pass: usize,
        context: &Context,
        views: TextureViews<'_>,
    );
}

#[derive(Debug)]
struct BlackBox<E: LayerEffect> {
    effect: E,
    passes: Vec<PassEntry<E>>,
    requirements: Requirements,
    expansion: Padding,
    translation_invariant: bool,
}

impl<E: LayerEffect> Stored for BlackBox<E> {
    fn effect_type_id(&self) -> TypeId {
        TypeId::of::<E>()
    }

    fn requirements(&self) -> Requirements {
        self.requirements
    }

    fn expansion(&self) -> Padding {
        self.expansion
    }

    fn contribute_inputs(&self, inputs: &mut LayerInputRecords) {
        self.effect.record_inputs(inputs);
    }

    fn is_translation_invariant(&self) -> bool {
        self.translation_invariant
    }

    fn passes_len(&self) -> usize {
        self.passes.len()
    }

    fn pass_requirements(&self, pass: usize) -> Requirements {
        self.passes[pass].requirements
    }

    fn pass_type_id(&self, pass: usize) -> TypeId {
        self.passes[pass].pass.pass_type_id()
    }

    fn contribute_pass_inputs(&self, pass: usize, inputs: &mut LayerInputRecords) {
        inputs.atoms.push(self.passes[pass].evidence.clone());
    }

    fn prepare(
        &self,
        storage: &mut Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        pass: usize,
        context: &Context,
        views: TextureViews<'_>,
    ) -> Box<dyn Erased> {
        let mut pipelines = PipelineRegistry {
            access: Access::Write(storage),
            device,
            queue,
            format,
        };
        self.passes[pass]
            .pass
            .prepare(&self.effect, &mut pipelines, device, queue, context, views)
    }

    fn encode(
        &self,
        storage: &Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        prepared: &dyn Erased,
        encoder: &mut wgpu::CommandEncoder,
        pass: usize,
        context: &Context,
        views: TextureViews<'_>,
    ) {
        let pipelines = PipelineRegistry {
            access: Access::Read(storage),
            device,
            queue,
            format,
        };
        self.passes[pass]
            .pass
            .encode(&self.effect, &pipelines, prepared, encoder, context, views);
    }
}

#[derive(Default)]
pub(crate) struct Storage {
    pipelines: HashMap<TypeId, Box<dyn Erased>>,
}

impl Storage {
    fn get_or_init<P: Pipeline>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> &mut P {
        self.pipelines
            .entry(TypeId::of::<P>())
            .or_insert_with(|| Box::new(P::new(device, queue, format)))
            .as_mut()
            .as_any_mut()
            .downcast_mut::<P>()
            .expect("isolated-layer effect pipeline type")
    }

    fn get<P: Pipeline>(&self) -> Option<&P> {
        self.pipelines
            .get(&TypeId::of::<P>())?
            .as_ref()
            .as_any()
            .downcast_ref::<P>()
    }
}

pub(crate) trait Erased: Any + MaybeSend + MaybeSync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T> Erased for T
where
    T: Any + MaybeSend + MaybeSync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub(crate) fn context(context: &super::Context, content_bounds: Rectangle) -> Context {
    Context {
        represented_bounds: context.represented_bounds,
        content_bounds,
        physical_size: context.physical_viewport(),
        backing_extent: context.backing_extent(),
        valid_uv: context.valid_uv(),
        scale_factor: context.scale_factor(),
        format: context.format,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

    #[derive(Debug, Clone, PartialEq)]
    struct First(u8);

    #[derive(Debug, Clone, PartialEq)]
    struct Second(u8);

    #[derive(Debug, Clone, PartialEq)]
    struct FloatingPointInputs {
        value_a: f32,
        value_b_two_dimensional: [f32; 2],
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestPass {
        id: usize,
        requirements: Requirements,
    }

    impl TestPass {
        fn new(id: usize) -> Self {
            Self {
                id,
                requirements: Requirements::new().writes_every_pixel(),
            }
        }
    }

    impl<E: LayerEffect> Pass<E> for TestPass {
        type Prepared = usize;

        fn requirements(&self, _effect: &E) -> Requirements {
            self.requirements
        }

        fn prepare(
            &self,
            _effect: &E,
            _pipelines: &mut PipelineRegistry<'_>,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _context: &Context,
            _views: TextureViews<'_>,
        ) -> Self::Prepared {
            self.id
        }

        fn encode(
            &self,
            _effect: &E,
            _pipelines: &PipelineRegistry<'_>,
            prepared: &Self::Prepared,
            _encoder: &mut wgpu::CommandEncoder,
            _context: &Context,
            _views: TextureViews<'_>,
        ) {
            assert_eq!(*prepared, self.id);
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AlternatePass(u16);

    impl<E: LayerEffect> Pass<E> for AlternatePass {
        type Prepared = u16;

        fn prepare(
            &self,
            _effect: &E,
            _pipelines: &mut PipelineRegistry<'_>,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _context: &Context,
            _views: TextureViews<'_>,
        ) -> Self::Prepared {
            self.0
        }

        fn encode(
            &self,
            _effect: &E,
            _pipelines: &PipelineRegistry<'_>,
            prepared: &Self::Prepared,
            _encoder: &mut wgpu::CommandEncoder,
            _context: &Context,
            _views: TextureViews<'_>,
        ) {
            assert_eq!(*prepared, self.0);
        }
    }

    macro_rules! test_effect {
        ($effect:ty) => {
            impl LayerEffect for $effect {
                fn plan(&self, plan: &mut Plan<'_, Self>) {
                    plan.push(TestPass::new(0));
                }

                fn is_translation_invariant(&self) -> bool {
                    true
                }
            }
        };
    }

    test_effect!(First);
    test_effect!(Second);
    test_effect!(FloatingPointInputs);

    #[derive(Debug, Clone, PartialEq)]
    struct Configurable {
        passes: usize,
        backdrop: bool,
        overwrites: bool,
        expansion: Padding,
        translation_invariant: bool,
    }

    impl LayerEffect for Configurable {
        fn plan(&self, plan: &mut Plan<'_, Self>) {
            for id in 0..self.passes {
                let requirements = if self.backdrop {
                    Requirements::new().with_backdrop()
                } else {
                    Requirements::new()
                };
                let requirements = if self.overwrites {
                    requirements.writes_every_pixel()
                } else {
                    requirements
                };
                plan.push(TestPass { id, requirements });
            }
        }

        fn expansion(&self) -> Padding {
            self.expansion
        }

        fn is_translation_invariant(&self) -> bool {
            self.translation_invariant
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct RevisionEffect(isolated_layer::ContentChangeHandle);

    impl LayerEffect for RevisionEffect {
        fn plan(&self, plan: &mut Plan<'_, Self>) {
            plan.push(TestPass {
                id: 0,
                requirements: Requirements::new(),
            });
        }

        fn record_inputs(&self, inputs: &mut LayerInputRecords) {
            inputs.depend_on(&self.0);
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct MixedPlan {
        reverse: bool,
        value: usize,
    }

    impl LayerEffect for MixedPlan {
        fn plan(&self, plan: &mut Plan<'_, Self>) {
            if self.reverse {
                plan.push(AlternatePass(9));
                plan.push(TestPass::new(self.value));
            } else {
                plan.push(TestPass::new(self.value));
                plan.push(AlternatePass(9));
            }
        }

        fn record_inputs(&self, inputs: &mut LayerInputRecords) {
            inputs.record(&());
        }
    }

    #[derive(Debug, Clone)]
    struct FrozenEffect {
        plan_calls: Arc<AtomicUsize>,
        passes: Arc<AtomicUsize>,
        backdrop: Arc<AtomicBool>,
        expansion_bits: Arc<AtomicU32>,
        translation_invariant: Arc<AtomicBool>,
    }

    impl PartialEq for FrozenEffect {
        fn eq(&self, other: &Self) -> bool {
            self.passes.load(Ordering::Relaxed) == other.passes.load(Ordering::Relaxed)
                && self.backdrop.load(Ordering::Relaxed) == other.backdrop.load(Ordering::Relaxed)
                && self.expansion_bits.load(Ordering::Relaxed)
                    == other.expansion_bits.load(Ordering::Relaxed)
                && self.translation_invariant.load(Ordering::Relaxed)
                    == other.translation_invariant.load(Ordering::Relaxed)
        }
    }

    impl LayerEffect for FrozenEffect {
        fn plan(&self, plan: &mut Plan<'_, Self>) {
            let _ = self.plan_calls.fetch_add(1, Ordering::Relaxed);
            let requirements = if self.backdrop.load(Ordering::Relaxed) {
                Requirements::new().with_backdrop()
            } else {
                Requirements::new()
            };

            for id in 0..self.passes.load(Ordering::Relaxed) {
                plan.push(TestPass { id, requirements });
            }
        }

        fn expansion(&self) -> Padding {
            Padding::new(f32::from_bits(self.expansion_bits.load(Ordering::Relaxed)))
        }

        fn record_inputs(&self, inputs: &mut LayerInputRecords) {
            inputs.record(&());
        }

        fn is_translation_invariant(&self) -> bool {
            self.translation_invariant.load(Ordering::Relaxed)
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct CompoundEffect;

    #[derive(Debug, Clone, PartialEq)]
    struct CompoundPass;

    struct CompoundPrepared {
        _textures: Vec<wgpu::Texture>,
    }

    impl LayerEffect for CompoundEffect {
        fn plan(&self, plan: &mut Plan<'_, Self>) {
            plan.push(CompoundPass);
        }
    }

    impl Pass<CompoundEffect> for CompoundPass {
        type Prepared = CompoundPrepared;

        fn prepare(
            &self,
            _effect: &CompoundEffect,
            _pipelines: &mut PipelineRegistry<'_>,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _context: &Context,
            _views: TextureViews<'_>,
        ) -> Self::Prepared {
            CompoundPrepared {
                _textures: Vec::new(),
            }
        }

        fn encode(
            &self,
            _effect: &CompoundEffect,
            _pipelines: &PipelineRegistry<'_>,
            _prepared: &Self::Prepared,
            encoder: &mut wgpu::CommandEncoder,
            _context: &Context,
            _views: TextureViews<'_>,
        ) {
            let _: &mut wgpu::CommandEncoder = encoder;
        }
    }

    #[test]
    fn requirements_default_to_no_backdrop_and_renderer_initialization() {
        let requirements = Requirements::new();

        assert_eq!(requirements, Requirements::default());
        assert!(!requirements.needs_backdrop());
        assert!(!requirements.fully_overwrites());
        assert!(requirements.with_backdrop().needs_backdrop());
        assert!(requirements.writes_every_pixel().fully_overwrites());
    }

    #[test]
    fn records_owned_clones_of_partial_eq_values() {
        let mut value = FloatingPointInputs {
            value_a: 0.123,
            value_b_two_dimensional: [1.0, 2.0],
        };
        let original = value.clone();

        let mut recorded = LayerInputRecords::new();
        recorded.record(&value);
        let recorded = recorded.finish();

        value.value_a = 9.0;

        let mut expected = LayerInputRecords::new();
        expected.record(&original);
        assert!(recorded.matches(&expected.finish()));

        let mut changed = LayerInputRecords::new();
        changed.record(&value);
        assert!(!recorded.matches(&changed.finish()));
    }

    #[test]
    fn layer_effect_records_its_value_by_default() {
        fn evidence(value_a: f32) -> LayerInputEvidence {
            EffectStack::from_effect(FloatingPointInputs {
                value_a,
                value_b_two_dimensional: [1.0, 2.0],
            })
            .input_evidence()
        }

        assert!(evidence(0.123).matches(&evidence(0.123)));
        assert!(!evidence(0.123).matches(&evidence(9.0)));
    }

    #[test]
    fn plan_structure_and_descriptor_values_are_automatic_evidence() {
        let baseline = EffectStack::from_effect(MixedPlan {
            reverse: false,
            value: 7,
        });
        let equivalent = EffectStack::from_effect(MixedPlan {
            reverse: false,
            value: 7,
        });
        let changed_value = EffectStack::from_effect(MixedPlan {
            reverse: false,
            value: 8,
        });
        let changed_order = EffectStack::from_effect(MixedPlan {
            reverse: true,
            value: 7,
        });

        assert!(
            baseline
                .input_evidence()
                .matches(&equivalent.input_evidence())
        );
        assert_ne!(baseline.input_evidence(), changed_value.input_evidence());
        assert_ne!(baseline.input_evidence(), changed_order.input_evidence());
        assert_eq!(baseline[0].passes_len(), 2);
        assert_eq!(
            baseline[0].stored().pass_type_id(0),
            TypeId::of::<TestPass>()
        );
        assert_eq!(
            baseline[0].stored().pass_type_id(1),
            TypeId::of::<AlternatePass>()
        );
    }

    #[test]
    fn compound_passes_can_own_textures_and_receive_a_raw_encoder() {
        let effect = Effect::new(CompoundEffect);

        assert_eq!(effect.passes_len(), 1);
        assert_eq!(
            effect.stored().pass_type_id(0),
            TypeId::of::<CompoundPass>()
        );
    }

    #[test]
    fn effect_new_freezes_the_plan_and_early_metadata_once() {
        let plan_calls = Arc::new(AtomicUsize::new(0));
        let passes = Arc::new(AtomicUsize::new(2));
        let backdrop = Arc::new(AtomicBool::new(true));
        let expansion_bits = Arc::new(AtomicU32::new(3.0f32.to_bits()));
        let translation_invariant = Arc::new(AtomicBool::new(true));
        let source = FrozenEffect {
            plan_calls: Arc::clone(&plan_calls),
            passes: Arc::clone(&passes),
            backdrop: Arc::clone(&backdrop),
            expansion_bits: Arc::clone(&expansion_bits),
            translation_invariant: Arc::clone(&translation_invariant),
        };
        let effect = Effect::new(source.clone());

        assert_eq!(plan_calls.load(Ordering::Relaxed), 1);
        assert_eq!(effect.passes_len(), 2);
        assert!(effect.requirements().needs_backdrop());
        assert_eq!(effect.expansion(), Padding::new(3.0));
        assert!(effect.is_translation_invariant());
        let frozen_stack = EffectStack::from_iter([effect.clone()]);
        let frozen_evidence = frozen_stack.input_evidence();

        passes.store(4, Ordering::Relaxed);
        backdrop.store(false, Ordering::Relaxed);
        expansion_bits.store(9.0f32.to_bits(), Ordering::Relaxed);
        translation_invariant.store(false, Ordering::Relaxed);

        assert!(frozen_evidence.matches(&frozen_stack.input_evidence()));

        let cloned = effect.clone();
        let stack = EffectStack::from_iter([effect, cloned]);
        let _ = stack.input_evidence();
        let _ = stack.recollect_input_evidence();
        assert_eq!(plan_calls.load(Ordering::Relaxed), 1);
        assert_eq!(stack[0].passes_len(), 2);
        assert!(stack[0].requirements().needs_backdrop());
        assert_eq!(stack[0].expansion(), Padding::new(3.0));
        assert!(stack[0].is_translation_invariant());

        let replacement = Effect::new(source);
        assert_eq!(plan_calls.load(Ordering::Relaxed), 2);
        assert_eq!(replacement.passes_len(), 4);
        assert!(!replacement.requirements().needs_backdrop());
        assert_eq!(replacement.expansion(), Padding::new(9.0));
        assert!(!replacement.is_translation_invariant());
    }

    #[test]
    fn empty_stage_retains_stage_metadata_without_requirements() {
        let stage = Configurable {
            passes: 0,
            backdrop: true,
            overwrites: true,
            expansion: Padding::new(4.0),
            translation_invariant: true,
        };
        let stack = EffectStack::from_effect(stage);

        assert_eq!(stack[0].passes_len(), 0);
        assert_eq!(stack[0].requirements(), Requirements::new());
        assert_eq!(stack.expansion(), Some(Padding::new(4.0)));
        assert!(stack.is_translation_invariant());
        assert_ne!(stack.input_evidence(), EffectStack::new().input_evidence());
    }

    #[test]
    fn evidence_uses_normal_floating_point_semantics() {
        fn evidence(value: f32) -> LayerInputEvidence {
            let mut inputs = LayerInputRecords::new();
            inputs.record(&value);
            inputs.finish()
        }

        assert!(evidence(0.0).matches(&evidence(-0.0)));

        let nan = evidence(f32::NAN);
        assert_ne!(nan, nan.clone());
        assert!(!nan.matches(&nan));
    }

    #[test]
    fn framework_evidence_distinguishes_type_order_and_boundaries() {
        let first_then_second = EffectStack::from_effect(First(7)).with(Second(9));
        let second_then_first = EffectStack::from_effect(Second(9)).with(First(7));
        let two_first = EffectStack::from_effect(First(7)).with(First(9));

        assert_ne!(
            first_then_second.input_evidence(),
            second_then_first.input_evidence()
        );
        assert_ne!(
            first_then_second.input_evidence(),
            two_first.input_evidence()
        );
    }

    #[test]
    fn framework_evidence_captures_frozen_stage_and_pass_metadata() {
        fn evidence(
            passes: usize,
            backdrop: bool,
            overwrites: bool,
            expansion: Padding,
            translation_invariant: bool,
        ) -> LayerInputEvidence {
            EffectStack::from_effect(Configurable {
                passes,
                backdrop,
                overwrites,
                expansion,
                translation_invariant,
            })
            .input_evidence()
        }

        let baseline = evidence(1, false, false, Padding::ZERO, false);

        assert_ne!(baseline, evidence(2, false, false, Padding::ZERO, false));
        assert_ne!(baseline, evidence(1, true, false, Padding::ZERO, false));
        assert_ne!(baseline, evidence(1, false, true, Padding::ZERO, false));
        assert_ne!(
            baseline,
            evidence(1, false, false, Padding::new(1.0), false)
        );
        assert_ne!(baseline, evidence(1, false, false, Padding::ZERO, true));
    }

    #[test]
    fn volatile_evidence_never_matches() {
        let mut inputs = LayerInputRecords::new();
        inputs.record(&());
        inputs.mark_volatile();
        let evidence = inputs.finish();

        assert!(evidence.is_volatile());
        assert_eq!(evidence, evidence.clone());
        assert!(!evidence.matches(&evidence));
    }

    #[test]
    fn stack_preserves_stage_requirements_and_aggregates_metadata() {
        let stack = EffectStack::from_effect(Configurable {
            passes: 32,
            backdrop: false,
            overwrites: true,
            expansion: Padding {
                top: 1.0,
                right: 2.0,
                bottom: 3.0,
                left: 4.0,
            },
            translation_invariant: true,
        })
        .with(Configurable {
            passes: 17,
            backdrop: true,
            overwrites: false,
            expansion: Padding {
                top: 4.0,
                right: 3.0,
                bottom: 2.0,
                left: 1.0,
            },
            translation_invariant: false,
        });

        let stages: Vec<_> = stack.stage_requirements().collect();
        assert_eq!(stages.len(), 2);
        assert_eq!(stack[0].passes_len(), 32);
        assert!(stages[0].fully_overwrites());
        assert_eq!(stack[1].passes_len(), 17);
        assert!(stages[1].needs_backdrop());
        assert!(!stages[1].fully_overwrites());

        let aggregate = stack.requirements();
        assert!(aggregate.needs_backdrop());
        assert!(!aggregate.fully_overwrites());
        assert_eq!(
            stack.expansion(),
            Some(Padding {
                top: 5.0,
                right: 5.0,
                bottom: 5.0,
                left: 5.0,
            })
        );
        assert!(!stack.is_translation_invariant());
    }

    #[test]
    fn stack_expansion_is_canonical_and_fallible() {
        assert_eq!(EffectStack::new().expansion(), Some(Padding::ZERO));

        let canonical = EffectStack::from_effect(Configurable {
            passes: 1,
            backdrop: false,
            overwrites: false,
            expansion: Padding {
                top: -0.0,
                right: 0.0,
                bottom: -0.0,
                left: 0.0,
            },
            translation_invariant: true,
        })
        .expansion()
        .expect("signed zero is a valid expansion");

        assert_eq!(canonical.top.to_bits(), 0.0f32.to_bits());
        assert_eq!(canonical.right.to_bits(), 0.0f32.to_bits());
        assert_eq!(canonical.bottom.to_bits(), 0.0f32.to_bits());
        assert_eq!(canonical.left.to_bits(), 0.0f32.to_bits());

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            let stack = EffectStack::from_effect(Configurable {
                passes: 1,
                backdrop: false,
                overwrites: false,
                expansion: Padding::new(invalid),
                translation_invariant: true,
            });

            assert!(stack.expansion().is_none());
        }

        let overflow = EffectStack::from_effect(Configurable {
            passes: 1,
            backdrop: false,
            overwrites: false,
            expansion: Padding::new(f32::MAX),
            translation_invariant: true,
        })
        .with(Configurable {
            passes: 1,
            backdrop: false,
            overwrites: false,
            expansion: Padding::new(f32::MAX),
            translation_invariant: true,
        });

        assert!(overflow.expansion().is_none());
    }

    #[test]
    fn canonical_expansion_metadata_normalizes_signed_zero() {
        let positive = EffectStack::from_effect(Configurable {
            passes: 1,
            backdrop: false,
            overwrites: false,
            expansion: Padding::ZERO,
            translation_invariant: true,
        });
        let negative = EffectStack::from_effect(Configurable {
            passes: 1,
            backdrop: false,
            overwrites: false,
            expansion: Padding::new(-0.0),
            translation_invariant: true,
        });

        assert_eq!(positive.input_evidence(), negative.input_evidence());
    }

    #[test]
    fn stack_recollects_from_the_same_erased_effect_instances() {
        let revision = isolated_layer::ContentChangeHandle::new();
        let stack = EffectStack::from_effect(RevisionEffect(revision.clone()));
        let recorded = stack.input_evidence();

        assert!(stack.inputs_match(&recorded));
        let _ = revision.mark_changed();
        assert!(!stack.inputs_match(&recorded));
    }

    #[test]
    fn translation_invariance_defaults_to_conservative() {
        let revision = isolated_layer::ContentChangeHandle::new();
        let stack = EffectStack::from_effect(RevisionEffect(revision));

        assert!(!stack.is_translation_invariant());
    }
}
