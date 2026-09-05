//! Capture widget subtrees in isolated layers and process them with WGPU.

use crate::core::isolated_layer as drawing;
use crate::core::layout::{self, Layout};
use crate::core::mouse;
use crate::core::overlay;
use crate::core::renderer;
use crate::core::widget::tree::{self, Tree};
use crate::core::widget::{Operation, Widget};
use crate::core::{Element, Event, Length, Rectangle, Shell, Size, Vector, window};

use crate::core::Padding;
use crate::renderer::wgpu::isolated_layer::effect as layer_effect;

mod cache;
mod effect;

pub use crate::renderer::wgpu::isolated_layer::effect::{
    Context, Effect, EffectStack, LayerEffect, LayerInputEvidence, LayerInputRecords, Pass,
    Pipeline, PipelineRegistry, Plan, Requirements, TextureViews,
};
use cache::CacheConfig;
pub use cache::CacheKeepAliveScope;
pub use effect::{AlphaMask, DropShadow, GaussianBlur};

/// A widget that captures its content in an isolated layer.
///
/// An ancestor viewport determines whether the layer is visible and clips the final composite.
/// Once visible, the complete expanded layer bounds are captured so retained pixels remain valid
/// if a later viewport exposes more of the same surface.
///
/// Effects appended with [`then_effect`](Self::then_effect) run in order. Every stage processes
/// the completed output of the preceding stage on one shared, aggregate final canvas. A later
/// stage is not given a separate view of the original child capture.
///
/// [`cache_output`](Self::cache_output) retains only the complete pre-composite output. Iced does
/// not currently expose descendant damage tracking, so cache validity for child pixels is a
/// caller-owned contract: share one or more
/// [`ContentChangeHandle`](drawing::ContentChangeHandle) values with application state or custom
/// child widgets, observe them with
/// [`observe_content_changes`](Self::observe_content_changes), and call
/// [`mark_changed`](drawing::ContentChangeHandle::mark_changed) whenever the corresponding pixels
/// may differ. Caching without any observed content handle fails closed.
///
/// Child pixels are considered translation invariant by default. Use
/// [`content_depends_on_translation(true)`](Self::content_depends_on_translation) if a custom
/// child primitive derives its pixels from absolute scene coordinates.
pub struct IsolatedLayer<'a, Message, Theme = crate::Theme, Renderer = crate::Renderer>
where
    Renderer: layer_effect::Renderer,
{
    content: Element<'a, Message, Theme, Renderer>,
    effects: layer_effect::EffectStack,
    composite: drawing::Composite,
    output_cache: Option<CacheConfig>,
    content_changes: Vec<drawing::ContentChangeHandle>,
    content_depends_on_translation: bool,
    cache_keep_alive_scope: CacheKeepAliveScope,
}

impl<'a, Message, Theme, Renderer> IsolatedLayer<'a, Message, Theme, Renderer>
where
    Renderer: layer_effect::Renderer,
{
    /// Creates an isolated layer around `content` with an empty effect stack.
    pub fn new(content: impl Into<Element<'a, Message, Theme, Renderer>>) -> Self {
        Self {
            content: content.into(),
            effects: layer_effect::EffectStack::new(),
            composite: drawing::Composite::default(),
            output_cache: None,
            content_changes: Vec::new(),
            content_depends_on_translation: false,
            cache_keep_alive_scope: CacheKeepAliveScope::default(),
        }
    }

    /// Creates an isolated layer around `content` with an effect applied for its
    /// rasterized pixels before compositing them onto the underlying parent layer.
    pub fn with_effect(
        content: impl Into<Element<'a, Message, Theme, Renderer>>,
        effect: impl layer_effect::LayerEffect,
    ) -> Self {
        Self::new(content).then_effect(effect)
    }

    /// Appends `effect` to the ordered effect stack.
    pub fn then_effect(mut self, effect: impl layer_effect::LayerEffect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Replaces the ordered effect stack.
    pub fn effects(mut self, effects: impl IntoIterator<Item = layer_effect::Effect>) -> Self {
        self.effects = effects.into_iter().collect();
        self
    }

    /// Sets how the effect output is composed into the parent.
    pub fn composite(mut self, composite: drawing::Composite) -> Self {
        self.composite = composite;
        self
    }

    /// Sets how the completed layer is positioned in its parent.
    pub fn positioning(mut self, positioning: drawing::CompositePositioning) -> Self {
        self.composite = self.composite.with_positioning(positioning);
        self
    }

    /// Adds caller-managed child-content change evidence.
    ///
    /// Share clones of `changes` with application state or custom child widgets and call
    /// [`ContentChangeHandle::mark_changed`](drawing::ContentChangeHandle::mark_changed) whenever
    /// their captured pixels may change. More than one handle may be observed.
    pub fn observe_content_changes(mut self, changes: &drawing::ContentChangeHandle) -> Self {
        self.content_changes.push(changes.clone());
        self
    }

    /// Adds several caller-managed child-content change handles.
    pub fn observe_content_changes_from<'b>(
        mut self,
        changes: impl IntoIterator<Item = &'b drawing::ContentChangeHandle>,
    ) -> Self {
        self.content_changes.extend(changes.into_iter().cloned());
        self
    }

    /// Declares whether captured child pixels depend on absolute translation.
    ///
    /// Child content is translation invariant by default. Set this to `true` when a custom child
    /// primitive observes absolute scene coordinates. Absolute geometry will then participate in
    /// output validity, so moving the layer causes a miss without manually marking content.
    pub fn content_depends_on_translation(mut self, depends: bool) -> Self {
        self.content_depends_on_translation = depends;
        self
    }

    /// Requests final-output caching with normal residency priority.
    ///
    /// At least one [`ContentChangeHandle`](drawing::ContentChangeHandle) must be observed before
    /// drawing. A request without content evidence fails closed and is not retained.
    pub fn cache_output(self, surface: &drawing::SurfaceHandle) -> Self {
        self.cache_output_with(surface, drawing::CacheResidencyPriority::Normal)
    }

    /// Requests final-output caching with protected residency priority.
    pub fn cache_output_protected(self, surface: &drawing::SurfaceHandle) -> Self {
        self.cache_output_with(surface, drawing::CacheResidencyPriority::Protected)
    }

    /// Requests final-output caching with an explicit residency priority.
    pub fn cache_output_with(
        mut self,
        surface: &drawing::SurfaceHandle,
        priority: drawing::CacheResidencyPriority,
    ) -> Self {
        self.output_cache = Some(CacheConfig::new(surface, priority));
        self
    }

    /// Sets the traversal policy used to keep the retained output resident.
    pub fn cache_keep_alive_scope(mut self, scope: CacheKeepAliveScope) -> Self {
        self.cache_keep_alive_scope = scope;

        self
    }

    /// Keeps retained pixels only while the producer is visibly recorded.
    ///
    /// This is the default cache keep-alive scope.
    pub fn visible_only(self) -> Self {
        self.cache_keep_alive_scope(CacheKeepAliveScope::VisibleOnly)
    }

    /// Also keeps retained pixels resident while redraw events visit this widget.
    ///
    /// This policy is best-effort: event capture or widget-specific routing can skip mounted
    /// subtrees. If visits stop and the producer is not visible, the cache may be swept and will
    /// be reconstructed when the producer draws again.
    pub fn keep_while_redraw_visited(self) -> Self {
        self.cache_keep_alive_scope(CacheKeepAliveScope::KeepWhileRedrawVisited)
    }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for IsolatedLayer<'_, Message, Theme, Renderer>
where
    Renderer: layer_effect::Renderer,
{
    fn tag(&self) -> tree::Tag {
        self.content.as_widget().tag()
    }

    fn state(&self) -> tree::State {
        self.content.as_widget().state()
    }

    fn diff(&mut self, tree: &mut Tree) {
        self.content.as_widget_mut().diff(tree);
    }

    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content.as_widget_mut().layout(tree, renderer, limits)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(tree, layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.content
            .as_widget_mut()
            .update(tree, event, layout, cursor, renderer, shell, viewport);

        if self.cache_keep_alive_scope == CacheKeepAliveScope::KeepWhileRedrawVisited
            && matches!(event, Event::Window(window::Event::RedrawRequested(_)))
            && let Some(config) = &self.output_cache
        {
            renderer.mark_cache_alive(config.keep_alive());
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content
            .as_widget()
            .mouse_interaction(tree, layout, cursor, viewport, renderer)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let content_bounds = layout.bounds();
        let Some(expansion) = self.effects.expansion() else {
            return;
        };
        let Some(bounds) = expanded(content_bounds, expansion) else {
            return;
        };
        let Some(clip) = bounds.intersection(viewport) else {
            return;
        };

        let mut layer = layer_effect::Layer::new(bounds, clip)
            .content_bounds(content_bounds)
            .composite(self.composite)
            .content_depends_on_translation(self.content_depends_on_translation);

        if let Some(config) = &self.output_cache {
            layer = layer.cache_output_with(
                &config.surface,
                config.priority,
                self.content_changes.iter(),
            );
        }

        renderer.with_isolated_layer_effects(layer, self.effects.clone(), |renderer| {
            self.content
                .as_widget()
                .draw(tree, renderer, theme, style, layout, cursor, &bounds);
        });
    }

    fn overlay<'a>(
        &'a mut self,
        tree: &'a mut Tree,
        layout: Layout<'a>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'a, Message, Theme, Renderer>> {
        self.content
            .as_widget_mut()
            .overlay(tree, layout, renderer, viewport, translation)
    }
}

impl<'a, Message, Theme, Renderer> From<IsolatedLayer<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Theme: 'a,
    Renderer: 'a + layer_effect::Renderer,
{
    fn from(layer: IsolatedLayer<'a, Message, Theme, Renderer>) -> Self {
        Element::new(layer)
    }
}

fn expanded(bounds: Rectangle, padding: Padding) -> Option<Rectangle> {
    if !valid_content_bounds(bounds) || !valid_expansion(padding) {
        return None;
    }

    let left = f64::from(bounds.x) - f64::from(padding.left);
    let top = f64::from(bounds.y) - f64::from(padding.top);
    let right = f64::from(bounds.x) + f64::from(bounds.width) + f64::from(padding.right);
    let bottom = f64::from(bounds.y) + f64::from(bounds.height) + f64::from(padding.bottom);
    let width = right - left;
    let height = bottom - top;

    let x = checked_geometry_value(left)?;
    let y = checked_geometry_value(top)?;
    let width = checked_geometry_value(width)?;
    let height = checked_geometry_value(height)?;
    let right = checked_geometry_value(right)?;
    let bottom = checked_geometry_value(bottom)?;

    if width <= 0.0
        || height <= 0.0
        || !(x + width).is_finite()
        || !(y + height).is_finite()
        || right < x
        || bottom < y
    {
        return None;
    }

    Some(Rectangle {
        x,
        y,
        width,
        height,
    })
}

fn valid_content_bounds(bounds: Rectangle) -> bool {
    bounds.x.is_finite()
        && bounds.y.is_finite()
        && bounds.width.is_finite()
        && bounds.height.is_finite()
        && bounds.width >= 0.0
        && bounds.height >= 0.0
}

fn valid_expansion(padding: Padding) -> bool {
    [padding.top, padding.right, padding.bottom, padding.left]
        .into_iter()
        .all(|side| side.is_finite() && side >= 0.0)
}

fn checked_geometry_value(value: f64) -> Option<f32> {
    if !value.is_finite() || value < -f64::from(f32::MAX) || value > f64::from(f32::MAX) {
        return None;
    }

    let value = value as f32;

    value
        .is_finite()
        .then_some(if value == 0.0 { 0.0 } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::Background;
    use crate::core::Point;
    use crate::core::Transformation;
    use crate::core::image;
    use crate::core::shell;

    use iced_runtime::user_interface;
    use std::cell::RefCell;

    #[test]
    fn expands_asymmetrically() {
        let bounds = Rectangle::new(crate::core::Point::new(10.0, 20.0), Size::new(30.0, 40.0));
        assert_eq!(
            expanded(
                bounds,
                Padding {
                    top: 1.0,
                    right: 2.0,
                    bottom: 3.0,
                    left: 4.0
                }
            ),
            Some(Rectangle::new(
                crate::core::Point::new(6.0, 19.0),
                Size::new(36.0, 44.0)
            ))
        );
    }

    #[test]
    fn expanded_bounds_reject_invalid_or_overflowing_geometry() {
        let bounds = Rectangle::new(Point::new(10.0, 20.0), Size::new(30.0, 40.0));

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert!(expanded(bounds, Padding::new(invalid)).is_none());
        }

        assert!(
            expanded(
                Rectangle::new(Point::ORIGIN, Size::new(f32::MAX, 1.0)),
                Padding::new(f32::MAX),
            )
            .is_none()
        );
        assert!(
            expanded(
                Rectangle::new(Point::new(f32::MAX, 0.0), Size::new(1.0, 1.0)),
                Padding::ZERO,
            )
            .is_none()
        );
    }

    #[test]
    fn positioning_defaults_to_snapped_and_survives_unrelated_builders() {
        let child_viewports = RefCell::new(Vec::new());
        let default: IsolatedLayer<'_, (), (), MockRenderer> =
            IsolatedLayer::new(crate::core::Element::new(ViewportProbe {
                viewports: &child_viewports,
            }));

        assert_eq!(
            default.composite.positioning(),
            drawing::CompositePositioning::Snapped
        );

        let surface = drawing::SurfaceHandle::new();
        let configured = default
            .composite(drawing::Composite::additive(0.25))
            .positioning(drawing::CompositePositioning::Subpixel)
            .cache_output(&surface)
            .cache_keep_alive_scope(CacheKeepAliveScope::KeepWhileRedrawVisited)
            .then_effect(ExpandingNoop);

        assert_eq!(configured.composite.opacity(), 0.25);
        assert_eq!(configured.composite.blend_mode(), drawing::BlendMode::Add);
        assert_eq!(
            configured.composite.positioning(),
            drawing::CompositePositioning::Subpixel
        );
        assert!(configured.output_cache.is_some());
        assert_eq!(
            configured.cache_keep_alive_scope,
            CacheKeepAliveScope::KeepWhileRedrawVisited
        );
    }

    #[test]
    fn heterogeneous_effects_share_one_nongeneric_layer_and_preserve_output_cache() {
        let surface = drawing::SurfaceHandle::new();
        let layer: IsolatedLayer<'_, (), (), MockRenderer> =
            IsolatedLayer::with_effect(crate::Space::new(), ExpandingNoop)
                .cache_output(&surface)
                .then_effect(GaussianBlur::new(2.0));

        assert!(layer.output_cache.is_some());
        assert_eq!(layer.effects.len(), 2);
    }

    #[test]
    fn child_content_is_translation_invariant_by_default_and_can_opt_in() {
        let default: IsolatedLayer<'_, (), (), MockRenderer> =
            IsolatedLayer::new(crate::Space::new());
        let dependent: IsolatedLayer<'_, (), (), MockRenderer> =
            IsolatedLayer::new(crate::Space::new()).content_depends_on_translation(true);

        assert!(!default.content_depends_on_translation);
        assert!(dependent.content_depends_on_translation);
    }

    #[test]
    fn partial_visibility_only_changes_the_final_layer_clip() {
        let child_viewports = RefCell::new(Vec::new());
        let layer: IsolatedLayer<'_, (), (), MockRenderer> = IsolatedLayer::with_effect(
            crate::core::Element::new(ViewportProbe {
                viewports: &child_viewports,
            }),
            ExpandingNoop,
        )
        .positioning(drawing::CompositePositioning::Subpixel);
        let node = layout::Node::new(Size::new(16.0, 16.0)).move_to(Point::new(10.0, 20.0));
        let bounds = expanded(
            node.bounds(),
            Padding {
                top: 4.0,
                right: 3.0,
                bottom: 5.0,
                left: 2.0,
            },
        )
        .expect("valid expanded bounds");
        let narrow_viewport = Rectangle::new(Point::new(0.0, 35.0), Size::new(100.0, 4.0));
        let wider_viewport = Rectangle::new(Point::new(0.0, 28.0), Size::new(100.0, 11.0));
        let tree = Tree::new(&layer as &dyn Widget<(), (), MockRenderer>);
        let mut renderer = MockRenderer::default();

        for viewport in [&narrow_viewport, &wider_viewport] {
            layer.draw(
                &tree,
                &mut renderer,
                &(),
                &renderer::Style::default(),
                Layout::new(&node),
                mouse::Cursor::Unavailable,
                viewport,
            );
        }

        assert_eq!(renderer.layers.len(), 2);
        assert_eq!(renderer.layers[0].bounds, bounds);
        assert_eq!(renderer.layers[1].bounds, bounds);
        assert_eq!(
            renderer.layers[0].clip,
            bounds.intersection(&narrow_viewport).unwrap()
        );
        assert_eq!(
            renderer.layers[1].clip,
            bounds.intersection(&wider_viewport).unwrap()
        );
        assert_eq!(&*child_viewports.borrow(), &[bounds, bounds]);
        assert!(renderer.layers.iter().all(|layer| {
            layer.composite.positioning() == drawing::CompositePositioning::Subpixel
        }));
    }

    #[test]
    fn fully_culled_layer_does_not_draw_its_child() {
        let child_viewports = RefCell::new(Vec::new());
        let layer: IsolatedLayer<'_, (), (), MockRenderer> =
            IsolatedLayer::new(crate::core::Element::new(ViewportProbe {
                viewports: &child_viewports,
            }));
        let node = layout::Node::new(Size::new(16.0, 16.0));
        let viewport = Rectangle::new(Point::new(0.0, 20.0), Size::new(16.0, 8.0));
        let tree = Tree::new(&layer as &dyn Widget<(), (), MockRenderer>);
        let mut renderer = MockRenderer::default();

        layer.draw(
            &tree,
            &mut renderer,
            &(),
            &renderer::Style::default(),
            Layout::new(&node),
            mouse::Cursor::Unavailable,
            &viewport,
        );

        assert!(renderer.layers.is_empty());
        assert!(child_viewports.borrow().is_empty());
    }

    #[test]
    fn invalid_effect_expansion_culls_before_recording_the_child() {
        let child_viewports = RefCell::new(Vec::new());
        let layer: IsolatedLayer<'_, (), (), MockRenderer> = IsolatedLayer::with_effect(
            crate::core::Element::new(ViewportProbe {
                viewports: &child_viewports,
            }),
            InvalidExpansion,
        );
        let node = layout::Node::new(Size::new(16.0, 16.0));
        let viewport = Rectangle::new(Point::ORIGIN, node.size());
        let tree = Tree::new(&layer as &dyn Widget<(), (), MockRenderer>);
        let mut renderer = MockRenderer::default();

        layer.draw(
            &tree,
            &mut renderer,
            &(),
            &renderer::Style::default(),
            Layout::new(&node),
            mouse::Cursor::Unavailable,
            &viewport,
        );

        assert!(renderer.layers.is_empty());
        assert!(child_viewports.borrow().is_empty());
    }

    #[test]
    fn cache_config_keeps_only_output_slot_residency() {
        let surface = drawing::SurfaceHandle::new();
        let config = CacheConfig::new(&surface, drawing::CacheResidencyPriority::Protected);
        let keep_alive = config.keep_alive();

        assert_eq!(keep_alive.identity(), surface.identity());
        assert_eq!(
            keep_alive.priority(),
            drawing::CacheResidencyPriority::Protected
        );
    }

    #[test]
    fn update_keep_alive_and_draw_use_separate_slot_and_content_handles() {
        let surface = drawing::SurfaceHandle::new();
        let first = drawing::ContentChangeHandle::new();
        let second = drawing::ContentChangeHandle::new();
        let mut layer: IsolatedLayer<'_, (), (), MockRenderer> = IsolatedLayer::new(
            crate::Space::new()
                .width(Length::Fixed(16.0))
                .height(Length::Fixed(16.0)),
        )
        // Scope builders deliberately work before caching is configured.
        .keep_while_redraw_visited()
        .observe_content_changes_from([&second, &first, &second])
        .cache_output_with(&surface, drawing::CacheResidencyPriority::Protected);

        let _ = first.mark_changed();

        let mut renderer = MockRenderer::default();
        update_and_draw(&mut layer, &mut renderer);

        let keep_alive = renderer.keep_alives.borrow()[0].clone();
        let cache_request = renderer.layers[0]
            .output_cache_request
            .as_ref()
            .expect("draw cache request");

        assert_eq!(keep_alive.identity(), cache_request.identity());
        assert_eq!(keep_alive.priority(), cache_request.priority());
        assert_eq!(
            keep_alive.priority(),
            drawing::CacheResidencyPriority::Protected
        );
        assert_eq!(
            cache_request.revisions(),
            &[first.revision(), second.revision()]
        );
    }

    #[test]
    fn visible_only_scope_never_emits_an_explicit_keep_alive() {
        let surface = drawing::SurfaceHandle::new();
        let content = drawing::ContentChangeHandle::new();
        let mut layer: IsolatedLayer<'_, (), (), MockRenderer> = IsolatedLayer::new(
            crate::Space::new()
                .width(Length::Fixed(16.0))
                .height(Length::Fixed(16.0)),
        )
        .keep_while_redraw_visited()
        .observe_content_changes(&content)
        .cache_output(&surface)
        .visible_only();

        let mut renderer = MockRenderer::default();
        update_and_draw(&mut layer, &mut renderer);

        assert!(renderer.keep_alives.borrow().is_empty());
        assert_eq!(renderer.layers.len(), 1);
    }

    #[test]
    fn output_cache_without_content_evidence_is_forwarded_fail_closed() {
        let surface = drawing::SurfaceHandle::new();
        let mut layer: IsolatedLayer<'_, (), (), MockRenderer> = IsolatedLayer::new(
            crate::Space::new()
                .width(Length::Fixed(16.0))
                .height(Length::Fixed(16.0)),
        )
        .cache_output(&surface);
        let mut renderer = MockRenderer::default();

        update_and_draw(&mut layer, &mut renderer);

        let request = renderer.layers[0]
            .output_cache_request
            .as_ref()
            .expect("output cache request");
        assert!(!request.has_content_evidence());
        assert!(!request.is_current());
    }

    #[test]
    fn keep_alive_scope_ignores_non_redraw_events() {
        let surface = drawing::SurfaceHandle::new();
        let content = drawing::ContentChangeHandle::new();
        let mut layer: IsolatedLayer<'_, (), (), MockRenderer> = IsolatedLayer::new(
            crate::Space::new()
                .width(Length::Fixed(16.0))
                .height(Length::Fixed(16.0)),
        )
        .observe_content_changes(&content)
        .cache_output(&surface)
        .keep_while_redraw_visited();
        let node = layout::Node::new(Size::new(16.0, 16.0));
        let viewport = Rectangle::new(Point::ORIGIN, node.size());
        let mut tree = Tree::new(&layer as &dyn Widget<(), (), MockRenderer>);
        let mut bus = shell::Bus::new();
        let mut shell = Shell::new(&window::Headless, shell::Waker::noop(), &mut bus);
        let renderer = MockRenderer::default();

        layer.update(
            &mut tree,
            &Event::Window(window::Event::CloseRequested),
            Layout::new(&node),
            mouse::Cursor::Unavailable,
            &renderer,
            &mut shell,
            &viewport,
        );

        assert!(renderer.keep_alives.borrow().is_empty());
    }

    #[test]
    fn runtime_overlay_routing_uses_update_keep_alives_or_visible_draw_fallback() {
        let (ignored_marks, ignored_layers, ignored_status) = run_overlay_frame(false);
        assert_eq!(ignored_status, crate::core::event::Status::Ignored);
        assert_eq!(ignored_marks, 1);
        assert_eq!(ignored_layers, 1);

        let (captured_marks, captured_layers, captured_status) = run_overlay_frame(true);
        assert_eq!(captured_status, crate::core::event::Status::Captured);
        assert_eq!(captured_marks, 0);
        assert_eq!(captured_layers, 1);
    }

    fn run_overlay_frame(capture: bool) -> (usize, usize, crate::core::event::Status) {
        let surface = drawing::SurfaceHandle::new();
        let content = drawing::ContentChangeHandle::new();
        let layer: IsolatedLayer<'_, (), (), MockRenderer> =
            IsolatedLayer::new(crate::core::Element::new(OverlayHost { capture }))
                .observe_content_changes(&content)
                .cache_output(&surface)
                .keep_while_redraw_visited();
        let mut renderer = MockRenderer::default();
        let mut interface = user_interface::UserInterface::build(
            layer,
            Size::new(16.0, 16.0),
            user_interface::Cache::new(),
            &mut renderer,
        );
        let mut messages = shell::Bus::new();
        let event = Event::Window(window::Event::RedrawRequested(
            crate::core::time::Instant::now(),
        ));
        let (_, statuses) = interface.update(
            &window::Headless,
            &shell::Waker::noop(),
            &[event],
            mouse::Cursor::Unavailable,
            &mut renderer,
            &mut messages,
        );

        interface.draw(
            &mut renderer,
            &(),
            &renderer::Style::default(),
            mouse::Cursor::Unavailable,
        );

        assert!(
            renderer.layers[0].output_cache_request.is_some(),
            "visible base draw must still record the retained producer",
        );

        (
            renderer.keep_alives.borrow().len(),
            renderer.layers.len(),
            statuses[0],
        )
    }

    #[test]
    fn output_slot_identity_is_independent_from_content_evidence() {
        let surface = drawing::SurfaceHandle::new();
        let content = drawing::ContentChangeHandle::new();
        let mut layer: IsolatedLayer<'_, (), (), MockRenderer> = IsolatedLayer::with_effect(
            crate::Space::new()
                .width(Length::Fixed(16.0))
                .height(Length::Fixed(16.0)),
            ExpandingNoop,
        )
        .observe_content_changes(&content)
        .cache_output_protected(&surface)
        .keep_while_redraw_visited();
        let _ = content.mark_changed();
        let mut renderer = MockRenderer::default();

        update_and_draw(&mut layer, &mut renderer);

        assert_eq!(renderer.keep_alives.borrow().len(), 1);
        assert_eq!(renderer.layers.len(), 1);
        let request = renderer.layers[0]
            .output_cache_request
            .as_ref()
            .expect("output cache request");
        assert_eq!(request.identity(), surface.identity());
        assert_eq!(
            request.priority(),
            drawing::CacheResidencyPriority::Protected
        );
        assert_eq!(request.revisions(), &[content.revision()]);
    }

    fn update_and_draw(
        layer: &mut IsolatedLayer<'_, (), (), MockRenderer>,
        renderer: &mut MockRenderer,
    ) {
        let node = layout::Node::new(Size::new(16.0, 16.0));
        let viewport = Rectangle::new(Point::ORIGIN, node.size());
        let mut tree = Tree::new(&*layer as &dyn Widget<(), (), MockRenderer>);
        let mut bus = shell::Bus::new();
        let mut shell = Shell::new(&window::Headless, shell::Waker::noop(), &mut bus);

        layer.update(
            &mut tree,
            &Event::Window(window::Event::RedrawRequested(
                crate::core::time::Instant::now(),
            )),
            Layout::new(&node),
            mouse::Cursor::Unavailable,
            renderer,
            &mut shell,
            &viewport,
        );
        layer.draw(
            &tree,
            renderer,
            &(),
            &renderer::Style::default(),
            Layout::new(&node),
            mouse::Cursor::Unavailable,
            &viewport,
        );
    }

    struct OverlayHost {
        capture: bool,
    }

    impl Widget<(), (), MockRenderer> for OverlayHost {
        fn size(&self) -> Size<Length> {
            Size::new(Length::Fixed(16.0), Length::Fixed(16.0))
        }

        fn layout(
            &mut self,
            _tree: &mut Tree,
            _renderer: &MockRenderer,
            _limits: &layout::Limits,
        ) -> layout::Node {
            layout::Node::new(Size::new(16.0, 16.0))
        }

        fn draw(
            &self,
            _tree: &Tree,
            _renderer: &mut MockRenderer,
            _theme: &(),
            _style: &renderer::Style,
            _layout: Layout<'_>,
            _cursor: mouse::Cursor,
            _viewport: &Rectangle,
        ) {
        }

        fn overlay<'a>(
            &'a mut self,
            _tree: &'a mut Tree,
            _layout: Layout<'a>,
            _renderer: &MockRenderer,
            _viewport: &Rectangle,
            _translation: Vector,
        ) -> Option<overlay::Element<'a, (), (), MockRenderer>> {
            Some(overlay::Element::new(Box::new(TestOverlay {
                capture: self.capture,
            })))
        }
    }

    struct TestOverlay {
        capture: bool,
    }

    impl overlay::Overlay<(), (), MockRenderer> for TestOverlay {
        fn layout(&mut self, _renderer: &MockRenderer, _bounds: Size) -> layout::Node {
            layout::Node::new(Size::new(16.0, 16.0))
        }

        fn draw(
            &self,
            _renderer: &mut MockRenderer,
            _theme: &(),
            _style: &renderer::Style,
            _layout: Layout<'_>,
            _cursor: mouse::Cursor,
        ) {
        }

        fn update(
            &mut self,
            _event: &Event,
            _layout: Layout<'_>,
            _cursor: mouse::Cursor,
            _renderer: &MockRenderer,
            shell: &mut Shell<'_, ()>,
        ) {
            if self.capture {
                shell.capture_event();
            }
        }
    }

    struct ViewportProbe<'a> {
        viewports: &'a RefCell<Vec<Rectangle>>,
    }

    impl Widget<(), (), MockRenderer> for ViewportProbe<'_> {
        fn size(&self) -> Size<Length> {
            Size::new(Length::Fixed(16.0), Length::Fixed(16.0))
        }

        fn layout(
            &mut self,
            _tree: &mut Tree,
            _renderer: &MockRenderer,
            _limits: &layout::Limits,
        ) -> layout::Node {
            layout::Node::new(Size::new(16.0, 16.0))
        }

        fn draw(
            &self,
            _tree: &Tree,
            _renderer: &mut MockRenderer,
            _theme: &(),
            _style: &renderer::Style,
            _layout: Layout<'_>,
            _cursor: mouse::Cursor,
            viewport: &Rectangle,
        ) {
            self.viewports.borrow_mut().push(*viewport);
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ExpandingNoop;

    impl layer_effect::LayerEffect for ExpandingNoop {
        fn plan(&self, plan: &mut layer_effect::Plan<'_, Self>) {
            plan.push(NoopPass);
        }

        fn expansion(&self) -> Padding {
            Padding {
                top: 4.0,
                right: 3.0,
                bottom: 5.0,
                left: 2.0,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct NoopPass;

    impl<E: layer_effect::LayerEffect> layer_effect::Pass<E> for NoopPass {
        type Prepared = ();

        fn prepare(
            &self,
            _effect: &E,
            _pipelines: &mut layer_effect::PipelineRegistry<'_>,
            _device: &crate::renderer::wgpu::wgpu::Device,
            _queue: &crate::renderer::wgpu::wgpu::Queue,
            _context: &layer_effect::Context,
            _inputs: layer_effect::TextureViews<'_>,
        ) {
        }

        fn encode(
            &self,
            _effect: &E,
            _pipelines: &layer_effect::PipelineRegistry<'_>,
            _prepared: &(),
            _encoder: &mut crate::renderer::wgpu::wgpu::CommandEncoder,
            _context: &layer_effect::Context,
            _inputs: layer_effect::TextureViews<'_>,
        ) {
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct InvalidExpansion;

    impl layer_effect::LayerEffect for InvalidExpansion {
        fn plan(&self, plan: &mut layer_effect::Plan<'_, Self>) {
            plan.push(NoopPass);
        }

        fn expansion(&self) -> Padding {
            Padding::new(f32::NAN)
        }
    }

    #[derive(Default)]
    struct MockRenderer {
        keep_alives: RefCell<Vec<drawing::CacheKeepAlive>>,
        layers: Vec<layer_effect::Layer>,
    }

    impl renderer::Renderer for MockRenderer {
        fn mark_cache_alive(&self, keep_alive: drawing::CacheKeepAlive) {
            self.keep_alives.borrow_mut().push(keep_alive);
        }

        fn start_layer(&mut self, _bounds: Rectangle) {}

        fn end_layer(&mut self) {}

        fn start_transformation(&mut self, _transformation: Transformation) {}

        fn end_transformation(&mut self) {}

        fn fill_quad(&mut self, _quad: renderer::Quad, _background: impl Into<Background>) {}

        fn allocate_image(
            &self,
            _handle: &image::Handle,
            _callback: impl FnOnce(Result<image::Allocation, image::Error>) + Send + 'static,
        ) {
        }

        fn hint(&mut self, _scale: renderer::Scale) {}

        fn scale(&self) -> Option<renderer::Scale> {
            None
        }

        fn reset(&mut self, _new_bounds: Rectangle) {}

        fn settings(&self) -> renderer::Settings {
            renderer::Settings::default()
        }
    }

    impl layer_effect::Renderer for MockRenderer {
        fn start_isolated_layer_effects(
            &mut self,
            layer: layer_effect::Layer,
            _effects: layer_effect::EffectStack,
        ) {
            self.layers.push(layer);
        }

        fn end_isolated_layer_effects(&mut self) {}
    }
}
