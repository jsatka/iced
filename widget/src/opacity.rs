//! Apply uniform group opacity to a widget subtree.

use crate::core::isolated_layer as drawing;
use crate::core::layout::{self, Layout};
use crate::core::mouse;
use crate::core::overlay;
use crate::core::renderer;
use crate::core::widget::tree::{self, Tree};
use crate::core::widget::{Operation, Widget};
use crate::core::{Element, Event, Length, Rectangle, Shell, Size, Vector};

/// A portable widget that applies uniform group opacity when the renderer supports isolation.
///
/// The WGPU renderer composites the captured content at the requested opacity. Renderers without
/// isolated-layer support, including tiny-skia, gracefully degrade by drawing the content
/// normally without applying group opacity. An ancestor viewport determines visibility and the
/// final composite clip; a visible group captures its complete bounds so later viewport changes
/// cannot expose missing source pixels.
pub struct Opacity<'a, Message, Theme = crate::Theme, Renderer = crate::Renderer>
where
    Renderer: crate::core::Renderer,
{
    content: Element<'a, Message, Theme, Renderer>,
    composite: drawing::Composite,
}

impl<'a, Message, Theme, Renderer> Opacity<'a, Message, Theme, Renderer>
where
    Renderer: crate::core::Renderer,
{
    /// Creates an opacity group around `content`.
    pub fn new(content: impl Into<Element<'a, Message, Theme, Renderer>>, opacity: f32) -> Self {
        Self {
            content: content.into(),
            composite: drawing::Composite::source_over(opacity),
        }
    }

    /// Sets how the completed opacity group is positioned in its parent.
    pub fn positioning(mut self, positioning: drawing::CompositePositioning) -> Self {
        self.composite = self.composite.with_positioning(positioning);
        self
    }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for Opacity<'_, Message, Theme, Renderer>
where
    Renderer: crate::core::Renderer,
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
        let bounds = layout.bounds();
        let Some(clip) = bounds.intersection(viewport) else {
            return;
        };
        let layer = drawing::Layer::new(bounds, clip).composite(self.composite);

        renderer.with_isolated_layer(layer, |renderer| {
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

impl<'a, Message, Theme, Renderer> From<Opacity<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Theme: 'a,
    Renderer: 'a + crate::core::Renderer,
{
    fn from(opacity: Opacity<'a, Message, Theme, Renderer>) -> Self {
        Element::new(opacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::image;
    use crate::core::{Background, Point, Transformation};

    use std::cell::RefCell;

    #[test]
    fn positioning_defaults_to_snapped_and_preserves_opacity() {
        let child_viewports = RefCell::new(Vec::new());
        let opacity: Opacity<'_, (), (), MockRenderer> = Opacity::new(
            crate::core::Element::new(ViewportProbe {
                viewports: &child_viewports,
            }),
            0.4,
        );

        assert_eq!(
            opacity.composite.positioning(),
            drawing::CompositePositioning::Snapped
        );

        let opacity = opacity.positioning(drawing::CompositePositioning::Subpixel);

        assert_eq!(opacity.composite.opacity(), 0.4);
        assert_eq!(
            opacity.composite.positioning(),
            drawing::CompositePositioning::Subpixel
        );
    }

    #[test]
    fn partial_visibility_only_changes_the_final_opacity_clip() {
        let child_viewports = RefCell::new(Vec::new());
        let opacity: Opacity<'_, (), (), MockRenderer> = Opacity::new(
            crate::core::Element::new(ViewportProbe {
                viewports: &child_viewports,
            }),
            0.5,
        )
        .positioning(drawing::CompositePositioning::Subpixel);
        let node = layout::Node::new(Size::new(16.0, 16.0)).move_to(Point::new(10.0, 20.0));
        let bounds = node.bounds();
        let narrow_viewport = Rectangle::new(Point::new(0.0, 32.0), Size::new(100.0, 4.0));
        let wider_viewport = Rectangle::new(Point::new(0.0, 26.0), Size::new(100.0, 10.0));
        let culled_viewport = Rectangle::new(Point::new(0.0, 40.0), Size::new(100.0, 4.0));
        let tree = Tree::new(&opacity as &dyn Widget<(), (), MockRenderer>);
        let mut renderer = MockRenderer::default();

        for viewport in [&narrow_viewport, &wider_viewport, &culled_viewport] {
            opacity.draw(
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

    #[derive(Default)]
    struct MockRenderer {
        layers: Vec<drawing::Layer>,
    }

    impl renderer::Renderer for MockRenderer {
        fn start_isolated_layer(&mut self, layer: drawing::Layer) {
            self.layers.push(layer);
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
}
