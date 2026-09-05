//! Declarative effect contracts for GPU texture-backed isolated layers.

use crate::core::{Padding, Rectangle, Size, isolated_layer};
use crate::graphics::futures::{MaybeSend, MaybeSync};

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::ops::Index;
use std::sync::Arc;

/// Declared resource requirements of an isolated-layer effect pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Requirements {
    passes: usize,
    backdrop: bool,
    writes_every_pixel: bool,
}

impl Requirements {
    /// Creates backdrop-free requirements for an ordered number of passes.
    ///
    /// The pass count is not clamped. Pass indices are local to the effect and
    /// use [`usize`] throughout the effect API.
    pub const fn passes(passes: usize) -> Self {
        Self {
            passes,
            backdrop: false,
            writes_every_pixel: false,
        }
    }

    /// Declares a parent-prefix backdrop input.
    pub const fn with_backdrop(mut self) -> Self {
        self.backdrop = true;
        self
    }

    /// Declares that the effect initializes every output pixel.
    pub const fn writes_every_pixel(mut self) -> Self {
        self.writes_every_pixel = true;
        self
    }

    /// Returns the declared pass count.
    pub const fn pass_count(self) -> usize {
        self.passes
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
    /// The output of the preceding local pass, or `stage_input` for pass zero.
    pub previous: &'a wgpu::TextureView,
    /// The dedicated output of this pass.
    pub output: &'a wgpu::TextureView,
}

/// Plain frame data describing one stage in an isolated-layer effect stack.
///
/// All pixel-affecting values observed through this trait—including requirements,
/// expansion, translation behavior, contributed inputs, and external resources—must
/// remain stable from input-evidence collection through preparation and encoding. A
/// mutable external resource must contribute stable identity plus a monotonic revision.
/// The renderer recollects evidence before retaining rendered pixels; this prevents a
/// changed candidate from entering the output cache, but it cannot make unrevisioned
/// interior mutation or an arbitrary change-and-revert sequence safe.
pub trait LayerEffect: Debug + Clone + PartialEq + MaybeSend + MaybeSync + 'static {
    /// Renderer-local prepared resources for one pass of this effect instance.
    type PreparedPass: Any + MaybeSend + MaybeSync;

    /// Declares all targets and parent-prefix dependencies before allocation.
    fn requirements(&self) -> Requirements;

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

    /// Prepares renderer-local resources for one declared pass.
    fn prepare_pass(
        &self,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: usize,
        context: &Context,
        views: TextureViews<'_>,
    ) -> Self::PreparedPass;

    /// Encodes one declared pass.
    ///
    /// `views.output` is dedicated to this pass and never aliases an input.
    /// Passes are encoded in increasing order, with pass numbers restarting at
    /// zero for each effect stage.
    fn encode_pass(
        &self,
        pipelines: &PipelineRegistry<'_>,
        prepared: &Self::PreparedPass,
        encoder: &mut wgpu::CommandEncoder,
        pass: usize,
        context: &Context,
        views: TextureViews<'_>,
    );
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
    /// Erases one concrete effect value while retaining it for recollection.
    pub fn new(effect: impl LayerEffect) -> Self {
        Self(Arc::new(BlackBox(effect)))
    }

    /// Returns the effect's declared pass and texture requirements.
    pub fn requirements(&self) -> Requirements {
        self.0.requirements()
    }

    /// Returns the effect's declared capture expansion.
    pub fn expansion(&self) -> Padding {
        self.0.expansion()
    }

    /// Contributes the effect-owned portion of its input evidence.
    pub fn contribute_inputs(&self, inputs: &mut LayerInputRecords) {
        self.0.contribute_inputs(inputs);
    }

    /// Returns whether the effect declares itself translation invariant.
    pub fn is_translation_invariant(&self) -> bool {
        self.0.is_translation_invariant()
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
    /// Pass counts are added without an isolated-layer-specific cap. Backdrop
    /// use is the union of the stages. Full overwrite is true only when every
    /// stage declares it and the stack is nonempty.
    pub fn requirements(&self) -> Requirements {
        let mut passes = 0;
        let mut backdrop = false;
        let mut writes_every_pixel = !self.effects.is_empty();

        for effect in &self.effects {
            let requirements = effect.requirements();
            passes += requirements.pass_count();
            backdrop |= requirements.needs_backdrop();
            writes_every_pixel &= requirements.fully_overwrites();
        }

        Requirements {
            passes,
            backdrop,
            writes_every_pixel,
        }
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
                requirements: effect.requirements(),
                expansion: expansion_bits(expansion),
                translation_invariant: effect.is_translation_invariant(),
            });
            effect.contribute_inputs(&mut inputs);
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

const INPUT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FrameworkInput {
    Stack {
        schema: u32,
        stages: usize,
    },
    Stage {
        index: usize,
        effect_type: TypeId,
        requirements: Requirements,
        expansion: [u32; 4],
        translation_invariant: bool,
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

pub(crate) trait Stored: Debug + MaybeSend + MaybeSync {
    fn effect_type_id(&self) -> TypeId;
    fn requirements(&self) -> Requirements;
    fn expansion(&self) -> Padding;
    fn contribute_inputs(&self, inputs: &mut LayerInputRecords);
    fn is_translation_invariant(&self) -> bool;

    fn prepare_pass(
        &self,
        storage: &mut Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        pass: usize,
        context: &Context,
        views: TextureViews<'_>,
    ) -> Box<dyn Erased>;

    fn encode_pass(
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
struct BlackBox<E: LayerEffect>(E);

impl<E: LayerEffect> Stored for BlackBox<E> {
    fn effect_type_id(&self) -> TypeId {
        TypeId::of::<E>()
    }

    fn requirements(&self) -> Requirements {
        self.0.requirements()
    }

    fn expansion(&self) -> Padding {
        self.0.expansion()
    }

    fn contribute_inputs(&self, inputs: &mut LayerInputRecords) {
        self.0.record_inputs(inputs);
    }

    fn is_translation_invariant(&self) -> bool {
        self.0.is_translation_invariant()
    }

    fn prepare_pass(
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
        Box::new(
            self.0
                .prepare_pass(&mut pipelines, device, queue, pass, context, views),
        )
    }

    fn encode_pass(
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
        let prepared = prepared
            .as_any()
            .downcast_ref::<E::PreparedPass>()
            .expect("isolated-layer prepared pass resources");
        self.0
            .encode_pass(&pipelines, prepared, encoder, pass, context, views);
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

    #[derive(Debug, Clone, PartialEq)]
    struct First(u8);

    #[derive(Debug, Clone, PartialEq)]
    struct Second(u8);

    #[derive(Debug, Clone, PartialEq)]
    struct FloatingPointInputs {
        value_a: f32,
        value_b_two_dimensional: [f32; 2],
    }

    macro_rules! test_effect {
        ($effect:ty) => {
            impl LayerEffect for $effect {
                type PreparedPass = ();

                fn requirements(&self) -> Requirements {
                    Requirements::passes(1).writes_every_pixel()
                }

                fn is_translation_invariant(&self) -> bool {
                    true
                }

                fn prepare_pass(
                    &self,
                    _pipelines: &mut PipelineRegistry<'_>,
                    _device: &wgpu::Device,
                    _queue: &wgpu::Queue,
                    _pass: usize,
                    _context: &Context,
                    _views: TextureViews<'_>,
                ) {
                }

                fn encode_pass(
                    &self,
                    _pipelines: &PipelineRegistry<'_>,
                    _prepared: &Self::PreparedPass,
                    _encoder: &mut wgpu::CommandEncoder,
                    _pass: usize,
                    _context: &Context,
                    _views: TextureViews<'_>,
                ) {
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
        type PreparedPass = ();

        fn requirements(&self) -> Requirements {
            let requirements = Requirements::passes(self.passes);
            let requirements = if self.backdrop {
                requirements.with_backdrop()
            } else {
                requirements
            };

            if self.overwrites {
                requirements.writes_every_pixel()
            } else {
                requirements
            }
        }

        fn expansion(&self) -> Padding {
            self.expansion
        }

        fn is_translation_invariant(&self) -> bool {
            self.translation_invariant
        }

        fn prepare_pass(
            &self,
            _pipelines: &mut PipelineRegistry<'_>,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _pass: usize,
            _context: &Context,
            _views: TextureViews<'_>,
        ) {
        }

        fn encode_pass(
            &self,
            _pipelines: &PipelineRegistry<'_>,
            _prepared: &Self::PreparedPass,
            _encoder: &mut wgpu::CommandEncoder,
            _pass: usize,
            _context: &Context,
            _views: TextureViews<'_>,
        ) {
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct RevisionEffect(isolated_layer::ContentChangeHandle);

    impl LayerEffect for RevisionEffect {
        type PreparedPass = ();

        fn requirements(&self) -> Requirements {
            Requirements::passes(1)
        }

        fn record_inputs(&self, inputs: &mut LayerInputRecords) {
            inputs.depend_on(&self.0);
        }

        fn prepare_pass(
            &self,
            _pipelines: &mut PipelineRegistry<'_>,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _pass: usize,
            _context: &Context,
            _views: TextureViews<'_>,
        ) {
        }

        fn encode_pass(
            &self,
            _pipelines: &PipelineRegistry<'_>,
            _prepared: &Self::PreparedPass,
            _encoder: &mut wgpu::CommandEncoder,
            _pass: usize,
            _context: &Context,
            _views: TextureViews<'_>,
        ) {
        }
    }

    #[test]
    fn pass_counts_are_not_clamped() {
        assert_eq!(Requirements::passes(usize::MAX).pass_count(), usize::MAX);
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
    fn framework_evidence_captures_declared_stage_metadata() {
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
        assert_eq!(stages[0].pass_count(), 32);
        assert!(stages[0].fully_overwrites());
        assert_eq!(stages[1].pass_count(), 17);
        assert!(stages[1].needs_backdrop());
        assert!(!stages[1].fully_overwrites());

        let aggregate = stack.requirements();
        assert_eq!(aggregate.pass_count(), 49);
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
