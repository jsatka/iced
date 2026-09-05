use crate::core::{Rectangle, Size, isolated_layer};
use crate::isolated_layer::effect;
use crate::layer;

/// An ordered sequence of direct leaves and isolated drawing nodes.
#[derive(Default)]
pub(crate) struct Sequence(pub Vec<Node>);

impl Sequence {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn needs_backdrop(&self) -> bool {
        self.0.iter().any(|node| match node {
            Node::Leaf(_) => false,
            Node::Layer(_) => false,
            Node::Effect(node) => node.effects.requirements().needs_backdrop(),
        })
    }
}

/// A node in an ordered drawing sequence.
pub(crate) enum Node {
    /// An uninterrupted ordinary WGPU layer stack.
    Leaf(Leaf),
    /// A bounded isolated layer.
    Layer(RecordedLayer),
    /// A bounded isolated-layer effect sequence.
    Effect(RecordedLayerEffect),
}

/// An uninterrupted ordinary WGPU layer stack.
pub(crate) struct Leaf {
    pub stack: layer::Stack,
    pub prepared: Vec<Option<PreparedLayer>>,
}

impl Leaf {
    fn new(stack: layer::Stack) -> Self {
        Self {
            stack,
            prepared: Vec::new(),
        }
    }
}

/// Prepared indices for one ordinary WGPU layer.
pub(crate) struct PreparedLayer {
    pub scissor: Rectangle<u32>,
    pub physical_bounds: Rectangle,
    pub quad: Option<usize>,
    pub triangle: std::ops::Range<usize>,
    #[cfg(any(feature = "svg", feature = "image"))]
    pub image: Option<usize>,
    pub text: std::ops::Range<usize>,
}

/// A bounded isolated layer node.
pub(crate) struct RecordedLayer {
    pub layer: isolated_layer::Layer,
    /// The requested size before the active renderer transform is applied.
    pub logical_surface_size: Size,
    pub content: Sequence,
    pub prepared: Option<super::PreparedIsolatedLayer>,
}

/// A bounded isolated-layer effect node.
pub(crate) struct RecordedLayerEffect {
    pub layer: effect::Layer,
    /// The requested size before the active renderer transform is applied.
    pub logical_surface_size: Size,
    pub effects: effect::EffectStack,
    pub content: Sequence,
    pub prepared: Option<super::PreparedLayerEffect>,
}

enum OpenKind {
    Plain(isolated_layer::Layer),
    Effect(effect::Layer, effect::EffectStack),
}

struct OpenLayer {
    kind: OpenKind,
    logical_surface_size: Size,
    content: Sequence,
    parent: layer::Stack,
}

/// Records exact ordering barriers while retaining the ordinary direct stack until needed.
#[derive(Default)]
pub(crate) struct Recorder {
    root: Sequence,
    open: Vec<OpenLayer>,
}

impl Recorder {
    pub fn is_segmented(&self) -> bool {
        !self.root.is_empty() || !self.open.is_empty()
    }

    pub fn start(&mut self, layers: &mut layer::Stack, mut layer: isolated_layer::Layer) {
        let bounds = layer.bounds;
        let transformation = layers.transformation();
        layer.bounds = layer.bounds * transformation;
        layer.clip = layer.clip * transformation;
        self.start_kind(layers, bounds, OpenKind::Plain(layer));
    }

    pub fn start_effects(
        &mut self,
        layers: &mut layer::Stack,
        mut layer: effect::Layer,
        effects: effect::EffectStack,
    ) {
        let bounds = layer.bounds;
        let transformation = layers.transformation();
        layer.bounds = layer.bounds * transformation;
        layer.content_bounds = layer.content_bounds * transformation;
        layer.clip = layer.clip * transformation;
        self.start_kind(layers, bounds, OpenKind::Effect(layer, effects));
    }

    fn start_kind(&mut self, layers: &mut layer::Stack, bounds: Rectangle, kind: OpenKind) {
        let before = layers.split();
        self.append_leaf(before);

        // A second split retains an empty parent continuation while leaving an equivalent empty
        // stack installed for the child capture.
        let parent = layers.split();
        layers.push_clip(bounds);

        self.open.push(OpenLayer {
            kind,
            logical_surface_size: bounds.size(),
            content: Sequence::default(),
            parent,
        });
    }

    pub fn end(&mut self, layers: &mut layer::Stack) {
        debug_assert!(!self.open.is_empty(), "unmatched isolated layer end");

        layers.pop_clip();
        let child = layers.split();
        self.append_leaf(child);

        let open = self.open.pop().expect("open isolated layer");
        *layers = open.parent;

        self.append(match open.kind {
            OpenKind::Plain(layer) => Node::Layer(RecordedLayer {
                layer,
                logical_surface_size: open.logical_surface_size,
                content: open.content,
                prepared: None,
            }),
            OpenKind::Effect(layer, effects) => Node::Effect(RecordedLayerEffect {
                layer,
                logical_surface_size: open.logical_surface_size,
                effects,
                content: open.content,
                prepared: None,
            }),
        });
    }

    pub fn take(&mut self, layers: &mut layer::Stack) -> Sequence {
        debug_assert!(self.open.is_empty(), "isolated layer scope left open");

        let tail = layers.split();
        self.append_leaf(tail);

        std::mem::take(&mut self.root)
    }

    /// Restores a completed sequence after its transient prepared state has been released.
    ///
    /// The renderer temporarily extracts the sequence while drawing to avoid aliasing the
    /// recorder with the rest of its mutable state. Restoring it keeps the recorded frame
    /// replayable for screenshots or any other draw performed before the next reset.
    pub fn restore(&mut self, sequence: Sequence) {
        debug_assert!(self.open.is_empty(), "isolated layer scope left open");
        debug_assert!(self.root.is_empty(), "recorded sequence was not extracted");

        self.root = sequence;
    }

    pub fn reset(&mut self) {
        debug_assert!(self.open.is_empty(), "isolated layer scope left open");
        self.root.0.clear();
    }

    fn append_leaf(&mut self, stack: layer::Stack) {
        if stack.iter().any(|layer| !layer.is_empty()) {
            self.append(Node::Leaf(Leaf::new(stack)));
        }
    }

    fn append(&mut self, node: Node) {
        if let Some(open) = self.open.last_mut() {
            open.content.0.push(node);
        } else {
            self.root.0.push(node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Background, Color, Point, Rectangle, Size, Transformation, renderer};

    fn draw_marker(stack: &mut layer::Stack, x: f32) {
        stack.current_mut().0.draw_quad(
            renderer::Quad {
                bounds: Rectangle::new(Point::new(x, 0.0), Size::new(1.0, 1.0)),
                ..renderer::Quad::default()
            },
            Background::Color(Color::WHITE),
            Transformation::IDENTITY,
        );
    }

    #[test]
    fn isolated_layer_is_an_exact_barrier() {
        let mut recorder = Recorder::default();
        let mut stack = layer::Stack::new();
        let bounds = Rectangle::new(Point::ORIGIN, Size::new(100.0, 100.0));

        draw_marker(&mut stack, 1.0);
        recorder.start(&mut stack, isolated_layer::Layer::new(bounds, bounds));
        draw_marker(&mut stack, 2.0);
        recorder.end(&mut stack);
        draw_marker(&mut stack, 3.0);

        let sequence = recorder.take(&mut stack);

        assert!(matches!(sequence.0[0], Node::Leaf(_)));
        assert!(matches!(sequence.0[1], Node::Layer(_)));
        assert!(matches!(sequence.0[2], Node::Leaf(_)));
    }

    #[test]
    fn nested_isolated_layers_are_recursive() {
        let mut recorder = Recorder::default();
        let mut stack = layer::Stack::new();
        let bounds = Rectangle::new(Point::ORIGIN, Size::new(100.0, 100.0));
        let layer = isolated_layer::Layer::new(bounds, bounds);

        recorder.start(&mut stack, layer.clone());
        draw_marker(&mut stack, 1.0);
        recorder.start(&mut stack, layer);
        draw_marker(&mut stack, 2.0);
        recorder.end(&mut stack);
        recorder.end(&mut stack);

        let sequence = recorder.take(&mut stack);
        let Node::Layer(outer) = &sequence.0[0] else {
            panic!("outer isolated layer");
        };

        assert!(matches!(outer.content.0[0], Node::Leaf(_)));
        assert!(matches!(outer.content.0[1], Node::Layer(_)));
    }

    #[test]
    fn isolated_layer_bounds_follow_the_active_transformation_once() {
        let mut recorder = Recorder::default();
        let mut stack = layer::Stack::new();
        let local = Rectangle::new(Point::new(2.0, 3.0), Size::new(10.0, 12.0));
        stack.push_transformation(Transformation::translate(20.0, 30.0));

        recorder.start(&mut stack, isolated_layer::Layer::new(local, local));
        draw_marker(&mut stack, 1.0);
        recorder.end(&mut stack);

        let sequence = recorder.take(&mut stack);
        let Node::Layer(node) = &sequence.0[0] else {
            panic!("isolated-layer node");
        };

        assert_eq!(
            node.layer.bounds,
            Rectangle::new(Point::new(22.0, 33.0), Size::new(10.0, 12.0))
        );
        assert_eq!(node.content.0.len(), 1);
    }
}
