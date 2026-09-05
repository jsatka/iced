//! Draw and stack layers of graphical primitives.
use crate::core::{Rectangle, Transformation};

/// A layer of graphical primitives.
///
/// Layers normally dictate a set of primitives that are
/// rendered in a specific order.
pub trait Layer: Default {
    /// Creates a new [`Layer`] with the given bounds.
    fn with_bounds(bounds: Rectangle) -> Self;

    /// Returns the current bounds of the [`Layer`].
    fn bounds(&self) -> Rectangle;

    /// Flushes and settles any pending group of primitives in the [`Layer`].
    ///
    /// This will be called when a [`Layer`] is finished. It allows layers to efficiently
    /// record primitives together and defer grouping until the end.
    fn flush(&mut self);

    /// Resizes the [`Layer`] to the given bounds.
    fn resize(&mut self, bounds: Rectangle);

    /// Clears all the layers contents and resets its bounds.
    fn reset(&mut self);

    /// Returns the start level of the [`Layer`].
    ///
    /// A level is a "sublayer" index inside of a [`Layer`].
    ///
    /// A [`Layer`] may draw multiple primitive types in a certain order.
    /// The level represents the lowest index of the primitive types it
    /// contains.
    ///
    /// Two layers A and B can therefore be merged if they have the same bounds,
    /// and the end level of A is lower or equal than the start level of B.
    fn start(&self) -> usize;

    /// Returns the end level of the [`Layer`].
    fn end(&self) -> usize;

    /// Merges a [`Layer`] with the current one.
    fn merge(&mut self, _layer: &mut Self);
}

/// A stack of layers used for drawing.
#[derive(Debug)]
pub struct Stack<T: Layer> {
    layers: Vec<T>,
    transformations: Vec<Transformation>,
    previous: Vec<usize>,
    current: usize,
    active_count: usize,
}

impl<T: Layer> Stack<T> {
    /// Creates a new empty [`Stack`].
    pub fn new() -> Self {
        Self {
            layers: vec![T::default()],
            transformations: vec![Transformation::IDENTITY],
            previous: vec![],
            current: 0,
            active_count: 1,
        }
    }

    /// Returns a mutable reference to the current [`Layer`] of the [`Stack`], together with
    /// the current [`Transformation`].
    #[inline]
    pub fn current_mut(&mut self) -> (&mut T, Transformation) {
        let transformation = self.transformation();

        (&mut self.layers[self.current], transformation)
    }

    /// Returns the current [`Transformation`] of the [`Stack`].
    #[inline]
    pub fn transformation(&self) -> Transformation {
        self.transformations.last().copied().unwrap()
    }

    /// Pushes a new clipping region in the [`Stack`]; creating a new layer in the
    /// process.
    pub fn push_clip(&mut self, bounds: Rectangle) {
        self.previous.push(self.current);

        self.current = self.active_count;
        self.active_count += 1;

        let bounds = bounds * self.transformation();

        if self.current == self.layers.len() {
            self.layers.push(T::with_bounds(bounds));
        } else {
            self.layers[self.current].resize(bounds);
        }
    }

    /// Pops the current clipping region from the [`Stack`] and restores the previous one.
    ///
    /// The current layer will be recorded for drawing.
    pub fn pop_clip(&mut self) {
        self.flush_current();

        self.current = self.previous.pop().unwrap();
    }

    /// Pushes a new [`Transformation`] in the [`Stack`].
    ///
    /// Future drawing operations will be affected by this new [`Transformation`] until
    /// it is popped using [`pop_transformation`].
    ///
    /// [`pop_transformation`]: Self::pop_transformation
    pub fn push_transformation(&mut self, transformation: Transformation) {
        self.transformations
            .push(self.transformation() * transformation);
    }

    /// Pops the current [`Transformation`] in the [`Stack`].
    pub fn pop_transformation(&mut self) {
        let _ = self.transformations.pop();
    }

    /// Returns an iterator over immutable references to the layers in the [`Stack`].
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.layers[..self.active_count].iter()
    }

    /// Returns the slice of layers in the [`Stack`].
    pub fn as_slice(&self) -> &[T] {
        &self.layers[..self.active_count]
    }

    /// Flushes and settles any primitives in every layer on the open clipping path.
    ///
    /// Layers that have already been popped are settled by [`pop_clip`](Self::pop_clip). The
    /// remaining ancestors and current layer are all settled so callers cannot accidentally
    /// observe or transfer a partially recorded stack.
    pub fn flush(&mut self) {
        for &index in &self.previous {
            self.layers[index].flush();
        }

        self.flush_current();
    }

    /// Flushes only the current layer when it is about to be closed.
    fn flush_current(&mut self) {
        self.layers[self.current].flush();
    }

    /// Splits the recorded layers at an ordering barrier.
    ///
    /// The returned stack contains everything recorded before the split. The current stack is
    /// replaced with an empty one that preserves the active clipping and transformation state, so
    /// matching [`pop_clip`](Self::pop_clip) and
    /// [`pop_transformation`](Self::pop_transformation) calls remain valid after the barrier. All
    /// layers in the open clipping path are flushed before the recorded prefix is returned.
    pub fn split(&mut self) -> Self {
        self.flush();

        let open_depth = self.previous.len();
        let layers = self
            .previous
            .iter()
            .copied()
            .chain(std::iter::once(self.current))
            .map(|index| T::with_bounds(self.layers[index].bounds()))
            .collect();

        let continuation = Self {
            layers,
            transformations: self.transformations.clone(),
            previous: (0..open_depth).collect(),
            current: open_depth,
            active_count: open_depth + 1,
        };

        std::mem::replace(self, continuation)
    }

    /// Performs layer merging wherever possible.
    ///
    /// Flushes and settles any primitives in the [`Stack`].
    pub fn merge(&mut self) {
        self.flush();

        // These are the layers left to process
        let mut left = self.active_count;

        // There must be at least 2 or more layers to merge
        while left > 1 {
            // We set our target as the topmost layer left to process
            let mut current = left - 1;
            let mut target = &self.layers[current];
            let mut target_start = target.start();
            let mut target_index = current;

            // We scan downwards for a contiguous block of mergeable layer candidates
            while current > 0 {
                current -= 1;

                let candidate = &self.layers[current];
                let start = candidate.start();
                let end = candidate.end();

                // We skip empty layers
                if end == 0 {
                    continue;
                }

                // Candidate can be merged if primitive sublayers do not overlap with
                // previous targets and the clipping bounds match
                if end > target_start || candidate.bounds() != target.bounds() {
                    break;
                }

                // Candidate is not empty and can be merged into
                target = candidate;
                target_start = start;
                target_index = current;
            }

            // We merge all the layers scanned into the target
            //
            // Since we use `target_index` instead of `current`, we
            // deliberately avoid merging into empty layers.
            //
            // If no candidates were mergeable, this is a no-op.
            let (head, tail) = self.layers.split_at_mut(target_index + 1);
            let layer = &mut head[target_index];

            for middle in &mut tail[0..left - target_index - 1] {
                layer.merge(middle);
            }

            // Empty layers found after the target can be skipped
            left = current;
        }
    }

    /// Clears the layers of the [`Stack`], allowing reuse.
    ///
    /// It resizes the base layer bounds to the `new_bounds`.
    ///
    /// This will normally keep layer allocations for future drawing operations.
    pub fn reset(&mut self, new_bounds: Rectangle) {
        for layer in self.layers[..self.active_count].iter_mut() {
            layer.reset();
        }

        self.layers[0].resize(new_bounds);
        self.current = 0;
        self.active_count = 1;
        self.previous.clear();
    }
}

impl<T: Layer> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct TestLayer {
        bounds: Rectangle,
        values: Vec<u8>,
        pending: Vec<u8>,
    }

    impl Layer for TestLayer {
        fn with_bounds(bounds: Rectangle) -> Self {
            Self {
                bounds,
                ..Self::default()
            }
        }

        fn bounds(&self) -> Rectangle {
            self.bounds
        }

        fn flush(&mut self) {
            self.values.append(&mut self.pending);
        }

        fn resize(&mut self, bounds: Rectangle) {
            self.bounds = bounds;
        }

        fn reset(&mut self) {
            self.values.clear();
            self.pending.clear();
        }

        fn start(&self) -> usize {
            usize::from(!self.values.is_empty())
        }

        fn end(&self) -> usize {
            usize::from(!self.values.is_empty())
        }

        fn merge(&mut self, layer: &mut Self) {
            self.values.append(&mut layer.values);
        }
    }

    #[test]
    fn split_preserves_active_recording_context() {
        let mut stack = Stack::<TestLayer>::new();
        let clip = Rectangle::new(
            crate::core::Point::new(10.0, 20.0),
            crate::core::Size::new(30.0, 40.0),
        );
        let translation = Transformation::translate(4.0, 8.0);

        stack.push_transformation(translation);
        stack.push_clip(clip);
        stack.current_mut().0.pending.push(1);

        let settled = stack.split();

        assert_eq!(
            settled
                .iter()
                .flat_map(|layer| &layer.values)
                .copied()
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert!(stack.iter().all(|layer| layer.values.is_empty()));
        assert_eq!(stack.transformation(), translation);

        stack.current_mut().0.pending.push(2);
        stack.pop_clip();
        stack.pop_transformation();

        assert_eq!(stack.transformation(), Transformation::IDENTITY);
        assert_eq!(
            stack
                .iter()
                .flat_map(|layer| &layer.values)
                .copied()
                .collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn split_settles_every_open_layer() {
        let mut stack = Stack::<TestLayer>::new();
        let outer = Rectangle::new(
            crate::core::Point::new(4.0, 6.0),
            crate::core::Size::new(80.0, 60.0),
        );
        let inner = Rectangle::new(
            crate::core::Point::new(8.0, 10.0),
            crate::core::Size::new(40.0, 30.0),
        );

        stack.current_mut().0.pending.push(1);
        stack.push_clip(outer);
        stack.current_mut().0.pending.push(2);
        stack.push_clip(inner);
        stack.current_mut().0.pending.push(3);

        let settled = stack.split();

        assert_eq!(
            settled
                .iter()
                .flat_map(|layer| &layer.values)
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(settled.iter().all(|layer| layer.pending.is_empty()));
        assert!(
            stack
                .iter()
                .all(|layer| { layer.values.is_empty() && layer.pending.is_empty() })
        );

        stack.current_mut().0.pending.push(4);
        stack.pop_clip();
        stack.current_mut().0.pending.push(5);
        stack.pop_clip();
        stack.current_mut().0.pending.push(6);
        stack.merge();

        assert_eq!(
            stack
                .iter()
                .flat_map(|layer| &layer.values)
                .copied()
                .collect::<Vec<_>>(),
            vec![6, 5, 4]
        );
    }

    #[test]
    fn public_flush_settles_every_open_layer() {
        let mut stack = Stack::<TestLayer>::new();
        let outer = Rectangle::new(
            crate::core::Point::new(4.0, 6.0),
            crate::core::Size::new(80.0, 60.0),
        );
        let inner = Rectangle::new(
            crate::core::Point::new(8.0, 10.0),
            crate::core::Size::new(40.0, 30.0),
        );

        stack.current_mut().0.pending.push(1);
        stack.push_clip(outer);
        stack.current_mut().0.pending.push(2);
        stack.push_clip(inner);
        stack.current_mut().0.pending.push(3);

        stack.flush();

        assert_eq!(
            stack
                .iter()
                .flat_map(|layer| &layer.values)
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(stack.iter().all(|layer| layer.pending.is_empty()));
    }

    #[test]
    fn split_compacts_the_empty_continuation_to_the_open_path() {
        let mut stack = Stack::<TestLayer>::new();
        let closed = Rectangle::new(
            crate::core::Point::new(1.0, 2.0),
            crate::core::Size::new(20.0, 20.0),
        );
        let outer = Rectangle::new(
            crate::core::Point::new(4.0, 6.0),
            crate::core::Size::new(80.0, 60.0),
        );
        let inner = Rectangle::new(
            crate::core::Point::new(8.0, 10.0),
            crate::core::Size::new(40.0, 30.0),
        );

        for value in 1..=8 {
            stack.push_clip(closed);
            stack.current_mut().0.pending.push(value);
            stack.pop_clip();
        }

        stack.push_clip(outer);
        stack.push_clip(inner);

        let prefix = stack.split();

        assert_eq!(prefix.iter().count(), 11);
        assert_eq!(stack.iter().count(), 3);
        assert_eq!(stack.current_mut().0.bounds, inner);

        stack.current_mut().0.pending.push(9);
        stack.pop_clip();
        assert_eq!(stack.current_mut().0.bounds, outer);
        stack.current_mut().0.pending.push(10);
        stack.pop_clip();
        stack.current_mut().0.pending.push(11);
        stack.merge();

        assert_eq!(
            stack
                .iter()
                .flat_map(|layer| &layer.values)
                .copied()
                .collect::<Vec<_>>(),
            vec![11, 10, 9]
        );
    }
}
