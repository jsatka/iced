use crate::core::{Point, Rectangle, Size, Transformation};
use crate::graphics::Viewport;

/// Selects the physical pixel grid used while capturing a bounded isolated layer.
///
/// A layer may be positioned at fractional physical coordinates in its immediate
/// parent target. This choice determines whether that fractional position is
/// resolved while rasterizing the layer, or preserved for its final composition.
///
/// This is a capture-geometry setting, not a final-placement policy. See
/// [`CompositePositioning`](crate::core::isolated_layer::CompositePositioning),
/// which determines whether the completed texture is placed on an integer pixel
/// boundary or at its exact fractional origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CaptureGrid {
    /// Capture into the immediate parent's physical-pixel grid.
    ///
    /// The layer's current fractional physical-pixel offset becomes part of the
    /// capture transform and is rasterized into its pixels. This can keep
    /// high-contrast details, such as text and one-pixel edges, sharper.
    ///
    /// As the captured pixel geometry will depend on the layer's absolute
    /// fractional pixel position, this is not recommended for a retained layer
    /// capture that may be recomposited in subsequents frames with different
    /// absolute fractional translations.
    ParentAligned,
    /// Capture into a physical-pixel grid local to the layer.
    ///
    /// The captured contents are rasterized relative to the requested layer
    /// origin. The exact fractional origin is preserved separately in [`Placement`].
    ///
    /// A snapped final composite may place this stable texture at an integer
    /// destination; a subpixel final composite can use that origin to linearly
    /// reconstruct it at its exact position.
    ///
    /// This is recommended for retained surfaces that may and enables smooth subpixel
    /// movement together with subpixel composition without rasterization artifacts.
    LayerAligned,
}

/// Final placement of a valid captured region in its immediate parent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Placement {
    snapped: Rectangle<u32>,
    exact_origin: Point,
}

impl Placement {
    fn root(size: Size<u32>) -> Self {
        Self {
            snapped: Rectangle::with_size(size),
            exact_origin: Point::ORIGIN,
        }
    }

    pub fn snapped(self) -> Rectangle<u32> {
        self.snapped
    }

    pub fn exact_origin(self) -> Point {
        self.exact_origin
    }

    pub fn exact_bounds(self) -> Rectangle {
        Rectangle::new(
            self.exact_origin,
            Size::new(self.snapped.width as f32, self.snapped.height as f32),
        )
    }

    /// Returns every parent pixel which can receive a non-zero filtered sample.
    pub fn conservative_coverage(self, parent: Size<u32>) -> Option<Rectangle<u32>> {
        let bounds = self.exact_bounds();

        intersect_parent(
            checked_floor(bounds.x)?,
            checked_floor(bounds.y)?,
            checked_ceil(bounds.x + bounds.width)?,
            checked_ceil(bounds.y + bounds.height)?,
            parent,
        )
    }
}

/// The complete coordinate and allocation context of an active raster target.
#[derive(Debug, Clone)]
pub(crate) struct Context {
    /// Bounds represented by the valid pixels of this target.
    ///
    /// These bounds use the same coordinate system as recorded layer and primitive bounds.
    /// Physical snapping and parent clipping may make them differ from the requested layer bounds.
    pub represented_bounds: Rectangle<f32>,
    /// The captured region expressed in the untrimmed layer's physical-pixel space.
    ///
    /// Its origin identifies the clipped offset into the layer, while its size is the valid
    /// viewport rendered into the target.
    pub source_rect: Rectangle<u32>,
    /// Final placement in the immediate parent target's pixel space.
    pub placement: Placement,
    /// The requested size before renderer transforms, stored as `f32` bits for retained validity.
    pub logical_surface_size_bits: [u32; 2],
    /// Unexpanded content geometry relative to `represented_bounds`, excluding absolute
    /// translation.
    pub source_content_relative_bits: [u32; 4],
    /// Full physical extent of the texture backing this target.
    ///
    /// Pooled textures may be larger than the valid viewport returned by
    /// [`Context::physical_viewport`].
    backing_extent: Size<u32>,
    /// Transformation from recorded bounds into this target's physical-pixel space.
    pub raster_transform: Transformation,
    /// Window and application scale decomposition inherited from the root viewport.
    ///
    /// `raster_transform` stores the total scale and target-local translation, but it cannot
    /// reconstruct the original decomposition expected by the public [`Viewport`].
    scale: crate::core::renderer::Scale,
    /// Format shared by the target and every compatible rendering pipeline.
    pub format: wgpu::TextureFormat,
}

impl Context {
    pub fn root(viewport: &Viewport, format: wgpu::TextureFormat) -> Self {
        let logical_size = viewport.logical_size();

        Self {
            represented_bounds: Rectangle::with_size(logical_size),
            source_rect: Rectangle::with_size(viewport.physical_size()),
            placement: Placement::root(viewport.physical_size()),
            logical_surface_size_bits: [
                logical_size.width.to_bits(),
                logical_size.height.to_bits(),
            ],
            source_content_relative_bits: [
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                logical_size.width.to_bits(),
                logical_size.height.to_bits(),
            ],
            backing_extent: viewport.physical_size(),
            raster_transform: Transformation::scale(viewport.scale_factor()),
            scale: viewport.scale(),
            format,
        }
    }

    /// Builds target-local geometry without choosing a pooled backing extent.
    pub fn bounded_with_grid(
        requested: Rectangle,
        parent: &Self,
        grid: CaptureGrid,
    ) -> Option<Self> {
        if !valid_rectangle(requested)
            || !parent.scale_factor().is_finite()
            || parent.scale_factor() <= 0.0
        {
            return None;
        }

        // Keep the intersection test separate from physical snapping. A tiny surface wholly outside
        // its parent must not appear merely because rounding moves it onto an edge pixel.
        let visible = requested.intersection(&parent.represented_bounds)?;

        let (placement, source_rect, raster_transform) = match grid {
            CaptureGrid::ParentAligned => parent_aligned_geometry(visible, parent)?,
            CaptureGrid::LayerAligned => layer_aligned_geometry(requested, parent)?,
        };
        let physical_viewport = Size::new(source_rect.width, source_rect.height);
        let represented_bounds = Rectangle::with_size(Size::new(
            physical_viewport.width as f32,
            physical_viewport.height as f32,
        )) * raster_transform.inverse();

        Some(Self {
            represented_bounds,
            source_rect,
            placement,
            logical_surface_size_bits: [requested.width.to_bits(), requested.height.to_bits()],
            source_content_relative_bits: relative_rectangle_bits(requested, represented_bounds),
            backing_extent: physical_viewport,
            raster_transform,
            scale: parent.scale,
            format: parent.format,
        })
    }

    /// Returns the valid physical-pixel viewport inside the backing texture.
    pub fn physical_viewport(&self) -> Size<u32> {
        Size::new(self.source_rect.width, self.source_rect.height)
    }

    /// Returns the full physical extent of the backing texture.
    pub fn backing_extent(&self) -> Size<u32> {
        self.backing_extent
    }

    /// Returns the scale encoded by the raster transform.
    pub fn scale_factor(&self) -> f32 {
        self.raster_transform.scale_factor()
    }

    /// Sets the backing extent after capture geometry has selected its valid viewport.
    pub fn set_backing_extent(&mut self, backing_extent: Size<u32>) {
        let physical_viewport = self.physical_viewport();

        debug_assert!(
            backing_extent.width >= physical_viewport.width
                && backing_extent.height >= physical_viewport.height,
            "GPU texture backing extent must contain its valid viewport"
        );

        self.backing_extent = backing_extent;
    }

    /// Records translation-invariant producer geometry for retained validity.
    pub fn set_source_geometry(&mut self, size: Size, content_bounds: Rectangle) {
        self.logical_surface_size_bits = [canonical_bits(size.width), canonical_bits(size.height)];
        self.source_content_relative_bits =
            relative_rectangle_bits(content_bounds, self.represented_bounds);
    }

    pub fn viewport(&self) -> Viewport {
        Viewport::with_physical_size(self.physical_viewport(), self.scale)
    }

    /// Transforms recorded bounds into target-local physical pixels without
    /// clipping them.
    pub fn raster_bounds(&self, bounds: Rectangle) -> Rectangle {
        bounds * self.raster_transform
    }

    pub fn local_bounds(&self, bounds: Rectangle) -> Option<Rectangle> {
        let physical_viewport = self.physical_viewport();
        let clipped = bounds.intersection(&self.represented_bounds)?;
        self.raster_bounds(clipped)
            .intersection(&Rectangle::with_size(Size::new(
                physical_viewport.width as f32,
                physical_viewport.height as f32,
            )))
    }

    pub fn local_scissor(&self, bounds: Rectangle) -> Option<Rectangle<u32>> {
        self.local_bounds(bounds)?.snap()
    }

    pub fn valid_uv(&self) -> [f32; 2] {
        let physical_viewport = self.physical_viewport();

        [
            physical_viewport.width as f32 / self.backing_extent.width as f32,
            physical_viewport.height as f32 / self.backing_extent.height as f32,
        ]
    }
}

fn parent_aligned_geometry(
    visible: Rectangle,
    parent: &Context,
) -> Option<(Placement, Rectangle<u32>, Transformation)> {
    let physical = visible * parent.raster_transform;
    let left = checked_floor(physical.x)?;
    let top = checked_floor(physical.y)?;
    let right = checked_ceil(physical.x + physical.width)?;
    let bottom = checked_ceil(physical.y + physical.height)?;
    let destination = intersect_parent(left, top, right, bottom, parent.physical_viewport())?;
    let source_rect = Rectangle::with_size(Size::new(destination.width, destination.height));
    let raster_transform =
        Transformation::translate(-(destination.x as f32), -(destination.y as f32))
            * parent.raster_transform;

    Some((
        Placement {
            snapped: destination,
            exact_origin: Point::new(destination.x as f32, destination.y as f32),
        },
        source_rect,
        raster_transform,
    ))
}

fn layer_aligned_geometry(
    requested: Rectangle,
    parent: &Context,
) -> Option<(Placement, Rectangle<u32>, Transformation)> {
    let physical = requested * parent.raster_transform;
    let surface_left = checked_round(physical.x)?;
    let surface_top = checked_round(physical.y)?;

    // The extent is intentionally independent of the absolute origin. Rounding the size in the
    // surface's own grid keeps both edges within half a physical pixel of the requested bounds,
    // without allowing fractional translation to resize the retained target and invalidate
    // otherwise-identical pixels.
    // A positive logical surface always owns at least one physical pixel. Keeping the
    // extent dependent only on size (never absolute edges) is what makes movement
    // descriptor-stable across fractional phases.
    let surface_width = checked_round(physical.width)?.max(1);
    let surface_height = checked_round(physical.height)?.max(1);

    let surface_right = surface_left.checked_add(surface_width)?;
    let surface_bottom = surface_top.checked_add(surface_height)?;
    let destination = intersect_parent(
        surface_left,
        surface_top,
        surface_right,
        surface_bottom,
        parent.physical_viewport(),
    )?;
    let source_x = u32::try_from(i64::from(destination.x).checked_sub(surface_left)?).ok()?;
    let source_y = u32::try_from(i64::from(destination.y).checked_sub(surface_top)?).ok()?;
    let source_rect = Rectangle {
        x: source_x,
        y: source_y,
        width: destination.width,
        height: destination.height,
    };
    let raster_transform = Transformation::translate(
        -(physical.x + source_x as f32),
        -(physical.y + source_y as f32),
    ) * parent.raster_transform;

    Some((
        Placement {
            snapped: destination,
            exact_origin: Point::new(physical.x + source_x as f32, physical.y + source_y as f32),
        },
        source_rect,
        raster_transform,
    ))
}

fn intersect_parent(
    left: i64,
    top: i64,
    right: i64,
    bottom: i64,
    parent: Size<u32>,
) -> Option<Rectangle<u32>> {
    let left = left.max(0).min(i64::from(parent.width));
    let top = top.max(0).min(i64::from(parent.height));
    let right = right.max(0).min(i64::from(parent.width));
    let bottom = bottom.max(0).min(i64::from(parent.height));

    if right <= left || bottom <= top {
        return None;
    }

    Some(Rectangle {
        x: u32::try_from(left).ok()?,
        y: u32::try_from(top).ok()?,
        width: u32::try_from(right.checked_sub(left)?).ok()?,
        height: u32::try_from(bottom.checked_sub(top)?).ok()?,
    })
}

fn checked_floor(value: f32) -> Option<i64> {
    checked_integer(value, f32::floor)
}

fn checked_ceil(value: f32) -> Option<i64> {
    checked_integer(value, f32::ceil)
}

fn checked_round(value: f32) -> Option<i64> {
    checked_integer(value, f32::round)
}

fn checked_integer(value: f32, operation: impl FnOnce(f32) -> f32) -> Option<i64> {
    const I64_MIN: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;

    let integer = f64::from(operation(value));
    (integer.is_finite() && (I64_MIN..I64_MAX_EXCLUSIVE).contains(&integer))
        .then_some(integer as i64)
}

fn valid_rectangle(rectangle: Rectangle) -> bool {
    rectangle.x.is_finite()
        && rectangle.y.is_finite()
        && rectangle.width.is_finite()
        && rectangle.height.is_finite()
        && rectangle.width > 0.0
        && rectangle.height > 0.0
}

fn canonical_bits(value: f32) -> u32 {
    if value == 0.0 {
        0.0f32.to_bits()
    } else {
        value.to_bits()
    }
}

fn relative_rectangle_bits(content: Rectangle, capture: Rectangle) -> [u32; 4] {
    [
        canonical_bits(content.x - capture.x),
        canonical_bits(content.y - capture.y),
        canonical_bits(content.width),
        canonical_bits(content.height),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(logical_size: Size, scale_factor: f32) -> Context {
        let physical_viewport = Size::new(
            (logical_size.width * scale_factor) as u32,
            (logical_size.height * scale_factor) as u32,
        );

        Context {
            represented_bounds: Rectangle::with_size(logical_size),
            source_rect: Rectangle::with_size(physical_viewport),
            placement: Placement::root(physical_viewport),
            logical_surface_size_bits: [
                logical_size.width.to_bits(),
                logical_size.height.to_bits(),
            ],
            source_content_relative_bits: [
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                logical_size.width.to_bits(),
                logical_size.height.to_bits(),
            ],
            backing_extent: physical_viewport,
            raster_transform: Transformation::scale(scale_factor),
            scale: crate::core::renderer::Scale {
                window: scale_factor,
                application: 1.0,
            },
            format: wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    fn assert_point_close(actual: Point, expected: Point) {
        assert!(
            (actual.x - expected.x).abs() < 0.001 && (actual.y - expected.y).abs() < 0.001,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn rectangle(x: u32, y: u32, width: u32, height: u32) -> Rectangle<u32> {
        Rectangle {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn transient_context_preserves_existing_localization() {
        let root = root(Size::new(800.0, 600.0), 2.0);

        let mut context = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.25, 20.25), Size::new(30.0, 40.0)),
            &root,
            CaptureGrid::ParentAligned,
        )
        .expect("bounded context");
        context.set_backing_extent(Size::new(64, 128));

        assert_eq!(context.placement.snapped(), rectangle(20, 40, 61, 81));
        assert_eq!(context.source_rect, Rectangle::with_size(Size::new(61, 81)));
        assert_eq!(context.physical_viewport(), Size::new(61, 81));
        assert_eq!(context.backing_extent(), Size::new(64, 128));
        assert_eq!(
            context.local_scissor(context.represented_bounds),
            Some(Rectangle::<u32> {
                x: 0,
                y: 0,
                width: 61,
                height: 81,
            })
        );
        assert_eq!(context.valid_uv(), [61.0 / 64.0, 81.0 / 128.0]);
    }

    #[test]
    fn retained_geometry_is_translation_invariant_across_pixel_phase() {
        let root = root(Size::new(100.0, 100.0), 2.0);
        let size = Size::new(30.2, 20.1);
        let first_requested = Rectangle::new(Point::new(10.2, 7.2), size);
        let second_requested = Rectangle::new(Point::new(10.3, 7.2), size);
        let first = Context::bounded_with_grid(first_requested, &root, CaptureGrid::LayerAligned)
            .expect("first retained context");
        let second = Context::bounded_with_grid(second_requested, &root, CaptureGrid::LayerAligned)
            .expect("second retained context");

        assert_eq!(first.source_rect, second.source_rect);
        assert_eq!(first.source_rect, Rectangle::with_size(Size::new(60, 40)));
        assert_eq!(
            first.logical_surface_size_bits,
            second.logical_surface_size_bits
        );
        assert_eq!(first.placement.snapped().x, 20);
        assert_eq!(second.placement.snapped().x, 21);
        assert_point_close(first.placement.exact_origin(), Point::new(20.4, 14.4));
        assert_point_close(second.placement.exact_origin(), Point::new(20.6, 14.4));
        assert_point_close(
            first_requested.position() * first.raster_transform,
            Point::ORIGIN,
        );
        assert_point_close(
            second_requested.position() * second.raster_transform,
            Point::ORIGIN,
        );
    }

    #[test]
    fn raster_bounds_preserve_uncut_geometry() {
        let root = root(Size::new(800.0, 600.0), 2.0);
        let mut context = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.25, 20.25), Size::new(30.0, 40.0)),
            &root,
            CaptureGrid::ParentAligned,
        )
        .expect("bounded context");
        context.set_backing_extent(Size::new(64, 128));

        assert_eq!(context.physical_viewport(), Size::new(61, 81));
        assert_eq!(context.backing_extent(), Size::new(64, 128));
        assert_eq!(
            context.raster_bounds(Rectangle::new(Point::new(5.0, 15.0), Size::new(20.0, 20.0),)),
            Rectangle::new(Point::new(-10.0, -10.0), Size::new(40.0, 40.0)),
        );
    }

    #[test]
    #[should_panic(expected = "GPU texture backing extent must contain its valid viewport")]
    fn backing_extent_must_contain_valid_viewport() {
        let mut context = root(Size::new(80.0, 60.0), 2.0);

        context.set_backing_extent(Size::new(159, 120));
    }

    #[test]
    fn layer_aligned_phase_is_derived_from_exact_and_snapped_origins() {
        let root = root(Size::new(100.0, 100.0), 1.0);
        let before = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.49, 8.25), Size::new(10.0, 10.0)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("before rounding boundary");
        let after = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.51, 8.25), Size::new(10.0, 10.0)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("after rounding boundary");

        let before_phase = before.placement.exact_origin().x - before.placement.snapped().x as f32;
        let after_phase = after.placement.exact_origin().x - after.placement.snapped().x as f32;

        assert!((before_phase - 0.49).abs() < 0.001);
        assert!((after_phase + 0.49).abs() < 0.001);
        assert_eq!(before.source_rect, after.source_rect);
    }

    #[test]
    fn conservative_coverage_contains_the_fractional_filter_footprint() {
        let placement = Placement {
            snapped: rectangle(11, 4, 10, 5),
            exact_origin: Point::new(10.75, 4.25),
        };

        assert_eq!(
            placement.conservative_coverage(Size::new(100, 100)),
            Some(rectangle(10, 4, 11, 6))
        );

        let clipped = Placement {
            snapped: rectangle(0, 0, 4, 4),
            exact_origin: Point::new(-0.25, -0.25),
        };
        assert_eq!(
            clipped.conservative_coverage(Size::new(100, 100)),
            Some(rectangle(0, 0, 4, 4))
        );
    }

    #[test]
    fn layer_aligned_transient_geometry_carries_phase_only_in_placement() {
        let root = root(Size::new(100.0, 100.0), 1.5);
        let bounds = Rectangle::new(Point::new(10.25, 8.5), Size::new(20.0, 10.0));
        let context = Context::bounded_with_grid(bounds, &root, CaptureGrid::LayerAligned)
            .expect("layer-aligned transient context");

        assert_point_close(bounds.position() * context.raster_transform, Point::ORIGIN);
        assert_point_close(context.placement.exact_origin(), Point::new(15.375, 12.75));
        assert_eq!(context.placement.snapped(), rectangle(15, 13, 30, 15));
    }

    #[test]
    fn retained_geometry_clips_negative_origins_with_signed_intermediates() {
        let root = root(Size::new(100.0, 100.0), 1.0);
        let requested = Rectangle::new(Point::new(-20.25, -10.25), Size::new(100.5, 50.5));
        let context = Context::bounded_with_grid(requested, &root, CaptureGrid::LayerAligned)
            .expect("clipped retained context");

        assert_eq!(context.placement.snapped(), rectangle(0, 0, 81, 41));
        assert_eq!(context.source_rect, rectangle(20, 10, 81, 41));
        assert_point_close(context.placement.exact_origin(), Point::new(-0.25, -0.25));
        assert_point_close(
            requested.position() * context.raster_transform,
            Point::new(-20.0, -10.0),
        );
    }

    #[test]
    fn retained_clipping_bands_with_equal_extents_do_not_alias() {
        let root = root(Size::new(50.0, 50.0), 1.0);
        let showing_bottom = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.0, -50.0), Size::new(20.0, 100.0)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("bottom band");
        let showing_top = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.0, 0.0), Size::new(20.0, 100.0)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("top band");

        assert_eq!(
            Size::new(
                showing_bottom.source_rect.width,
                showing_bottom.source_rect.height
            ),
            Size::new(
                showing_top.source_rect.width,
                showing_top.source_rect.height
            )
        );
        assert_eq!(showing_bottom.source_rect.y, 50);
        assert_eq!(showing_top.source_rect.y, 0);
        assert_ne!(showing_bottom.source_rect, showing_top.source_rect);
    }

    #[test]
    fn nested_retained_geometry_uses_parent_target_coordinates() {
        let root = root(Size::new(100.0, 100.0), 1.0);
        let outer_size = Size::new(60.2, 60.2);
        let inner_size = Size::new(20.2, 20.2);
        let outer_first_bounds = Rectangle::new(Point::new(10.4, 5.0), outer_size);
        let outer_second_bounds = Rectangle::new(Point::new(10.7, 5.0), outer_size);
        let outer_first =
            Context::bounded_with_grid(outer_first_bounds, &root, CaptureGrid::LayerAligned)
                .expect("first outer context");
        let outer_second =
            Context::bounded_with_grid(outer_second_bounds, &root, CaptureGrid::LayerAligned)
                .expect("second outer context");
        let inner_first_bounds = Rectangle::new(Point::new(20.4, 15.0), inner_size);
        let inner_second_bounds = Rectangle::new(Point::new(20.7, 15.0), inner_size);
        let inner_first =
            Context::bounded_with_grid(inner_first_bounds, &outer_first, CaptureGrid::LayerAligned)
                .expect("first inner context");
        let inner_second = Context::bounded_with_grid(
            inner_second_bounds,
            &outer_second,
            CaptureGrid::LayerAligned,
        )
        .expect("second inner context");

        assert_eq!(
            inner_first.placement.snapped(),
            inner_second.placement.snapped()
        );
        assert_eq!(inner_first.source_rect, inner_second.source_rect);
        assert_eq!(inner_first.placement.snapped().x, 10);
        assert_point_close(
            Point::new(
                outer_first.placement.exact_origin().x + inner_first.placement.exact_origin().x,
                outer_first.placement.exact_origin().y + inner_first.placement.exact_origin().y,
            ),
            inner_first_bounds.position(),
        );
        assert_point_close(
            Point::new(
                outer_second.placement.exact_origin().x + inner_second.placement.exact_origin().x,
                outer_second.placement.exact_origin().y + inner_second.placement.exact_origin().y,
            ),
            inner_second_bounds.position(),
        );
        assert_point_close(
            inner_first_bounds.position() * inner_first.raster_transform,
            Point::ORIGIN,
        );
        assert_point_close(
            inner_second_bounds.position() * inner_second.raster_transform,
            Point::ORIGIN,
        );
    }

    #[test]
    fn transient_child_preserves_phase_inside_a_retained_parent() {
        let root = root(Size::new(100.0, 100.0), 1.0);
        let outer_first = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.4, 5.0), Size::new(60.2, 60.2)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("first outer context");
        let outer_second = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.7, 5.0), Size::new(60.2, 60.2)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("second outer context");
        let inner_first_bounds = Rectangle::new(Point::new(20.65, 15.0), Size::new(10.2, 10.2));
        let inner_second_bounds = Rectangle::new(Point::new(20.95, 15.0), Size::new(10.2, 10.2));
        let inner_first = Context::bounded_with_grid(
            inner_first_bounds,
            &outer_first,
            CaptureGrid::ParentAligned,
        )
        .expect("first inner context");
        let inner_second = Context::bounded_with_grid(
            inner_second_bounds,
            &outer_second,
            CaptureGrid::ParentAligned,
        )
        .expect("second inner context");

        assert_eq!(
            inner_first.placement.snapped(),
            inner_second.placement.snapped()
        );
        assert_eq!(inner_first.source_rect, inner_second.source_rect);
        assert_point_close(
            inner_first_bounds.position() * inner_first.raster_transform,
            Point::new(0.25, 0.0),
        );
        assert_point_close(
            inner_second_bounds.position() * inner_second.raster_transform,
            Point::new(0.25, 0.0),
        );
    }

    #[test]
    fn logical_size_bits_distinguish_sizes_with_the_same_physical_extent() {
        let root = root(Size::new(300.0, 100.0), 1.0);
        let first = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.0, 10.0), Size::new(100.1, 20.0)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("first size");
        let second = Context::bounded_with_grid(
            Rectangle::new(Point::new(10.0, 10.0), Size::new(100.4, 20.0)),
            &root,
            CaptureGrid::LayerAligned,
        )
        .expect("second size");

        assert_eq!(first.source_rect, second.source_rect);
        assert_ne!(
            first.logical_surface_size_bits,
            second.logical_surface_size_bits
        );
    }

    #[test]
    fn invalid_or_nonintersecting_bounds_are_rejected_before_snapping() {
        let root = root(Size::new(100.0, 100.0), 1.0);

        assert!(
            Context::bounded_with_grid(
                Rectangle::new(Point::new(f32::INFINITY, 0.0), Size::new(10.0, 10.0)),
                &root,
                CaptureGrid::LayerAligned,
            )
            .is_none()
        );
        assert!(
            Context::bounded_with_grid(
                Rectangle::new(Point::new(-0.4, 10.0), Size::new(0.1, 10.0)),
                &root,
                CaptureGrid::LayerAligned,
            )
            .is_none()
        );
    }
}
