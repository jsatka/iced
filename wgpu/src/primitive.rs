//! Draw custom WGPU primitives.
//!
//! Prefer [`Primitive::draw`] for a normal draw call. Its render pass already
//! has the primitive viewport and scissor configured. Use [`Primitive::render`]
//! for work which needs its own pass or encoder commands.
use crate::core::{self, Rectangle, Size, Transformation};
use crate::graphics::Viewport;
use crate::graphics::futures::{MaybeSend, MaybeSync};

use rustc_hash::FxHashMap;
use std::any::{Any, TypeId};
use std::fmt::Debug;

/// A batch of primitives.
pub type Batch = Vec<Instance>;

/// Describes a primitive occurrence and the render target for which it is being
/// prepared.
///
/// [`Self::bounds`] preserves the scene-logical coordinate system used when the
/// primitive was recorded.
///
/// The remaining methods describe the target's valid viewport, backing-texture
/// extent, and mapping from scene-logical coordinates to target-local physical
/// pixels. An isolated or intermediate target may have a valid viewport smaller
/// than its pooled backing texture.
#[derive(Debug, Clone)]
pub struct PrepareRegion {
    bounds: Rectangle,
    viewport: Viewport,
    represented_bounds: Rectangle,
    backing_extent: Size<u32>,
    raster_transform: Transformation,
}

impl PrepareRegion {
    pub(crate) fn new(
        bounds: Rectangle,
        viewport: &Viewport,
        represented_bounds: Rectangle,
        backing_extent: Size<u32>,
        raster_transform: Transformation,
    ) -> Self {
        Self {
            bounds,
            viewport: viewport.clone(),
            represented_bounds,
            backing_extent,
            raster_transform,
        }
    }

    /// Returns the complete, uncut primitive bounds in scene-logical pixels.
    ///
    /// Active transformations have already been applied. These bounds retain
    /// the same coordinate system and semantics as the `bounds` argument
    /// supplied to `Primitive::prepare` before isolated-layer support.
    pub fn bounds(&self) -> &Rectangle {
        &self.bounds
    }

    /// Returns the valid viewport of the active render target.
    ///
    /// Its physical size equals [`Self::backing_extent`] for a direct root
    /// target and whenever an offscreen allocation exactly fits its valid
    /// region. A pooled offscreen target—such as an isolated target or an
    /// intermediate root target—may have a larger backing extent; the viewport
    /// excludes this unused padding.
    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    /// Returns the scene-logical bounds represented by the valid target pixels.
    ///
    /// This rectangle uses the same coordinate system as [`Self::bounds`].
    /// Physical snapping and ancestor clipping can make it differ from the
    /// originally requested isolated-layer bounds.
    pub fn represented_bounds(&self) -> Rectangle {
        self.represented_bounds
    }

    /// Returns the full physical extent of the backing color texture.
    ///
    /// Use this extent for depth, stencil, or other attachments which must
    /// match the color attachment. It can be larger than
    /// `self.viewport().physical_size()` when an offscreen target uses a pooled
    /// allocation.
    pub fn backing_extent(&self) -> Size<u32> {
        self.backing_extent
    }

    /// Returns the primitive bounds in target-local logical pixels.
    ///
    /// The result has the same logical size as [`Self::bounds`], with its origin
    /// translated so the top-left of [`Self::represented_bounds`] is `(0, 0)`.
    pub fn local_bounds(&self) -> Rectangle {
        Rectangle {
            x: self.bounds.x - self.represented_bounds.x,
            y: self.bounds.y - self.represented_bounds.y,
            ..self.bounds
        }
    }

    /// Returns the complete, uncut primitive bounds in target-local physical
    /// pixels.
    pub fn physical_bounds(&self) -> Rectangle {
        self.bounds * self.raster_transform
    }

    /// Returns the transformation from scene-logical coordinates to
    /// target-local physical pixels.
    pub fn logical_to_physical(&self) -> Transformation {
        self.raster_transform
    }

    /// Returns the transformation from scene-logical coordinates to normalized
    /// device coordinates for the active target.
    pub fn logical_projection(&self) -> Transformation {
        self.viewport.projection() * self.raster_transform
    }
}

/// The authoritative region used to render one [`Primitive`] occurrence.
///
/// The physical viewport remains complete and uncut so clipping cannot stretch
/// primitive-local normalized device coordinates. The physical scissor is the
/// visible intersection with the active Iced clip and valid target pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderRegion {
    bounds: Rectangle,
    physical_viewport: Rectangle,
    physical_scissor: Rectangle<u32>,
}

impl RenderRegion {
    pub(crate) fn new(
        bounds: Rectangle,
        physical_viewport: Rectangle,
        physical_scissor: Rectangle<u32>,
    ) -> Self {
        Self {
            bounds,
            physical_viewport,
            physical_scissor,
        }
    }

    /// Returns the complete, uncut primitive bounds in scene-logical pixels.
    pub fn bounds(&self) -> Rectangle {
        self.bounds
    }

    /// Returns the complete, uncut WGPU viewport in target-local physical
    /// pixels.
    pub fn physical_viewport(&self) -> Rectangle {
        self.physical_viewport
    }

    /// Returns the WGPU scissor rectangle in target-local physical pixels.
    pub fn physical_scissor(&self) -> Rectangle<u32> {
        self.physical_scissor
    }

    /// Configures a render pass with this primitive's physical viewport and
    /// scissor rectangle.
    pub fn configure(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_viewport(
            self.physical_viewport.x,
            self.physical_viewport.y,
            self.physical_viewport.width,
            self.physical_viewport.height,
            0.0,
            1.0,
        );
        render_pass.set_scissor_rect(
            self.physical_scissor.x,
            self.physical_scissor.y,
            self.physical_scissor.width,
            self.physical_scissor.height,
        );
    }
}

/// A set of methods which allows a [`Primitive`] to be rendered.
pub trait Primitive: Debug + MaybeSend + MaybeSync + 'static {
    /// The shared pipeline of this [`Primitive`].
    ///
    /// Normally, this will contain a bunch of [`wgpu`] state; like
    /// a rendering pipeline, buffers, and textures.
    ///
    /// All instances of this [`Primitive`] type will share the same
    /// [`Pipeline`].
    type Pipeline: Pipeline + MaybeSend + MaybeSync;

    /// Processes the [`Primitive`], allowing for GPU buffer allocation.
    ///
    /// The pipeline is shared by all occurrences of this primitive type, and
    /// the renderer may prepare several occurrences or targets before rendering
    /// any of them. An implementation which needs distinct resources in this
    /// situation must manage and select them in the pipeline itself.
    ///
    /// [`PrepareRegion::bounds`] retains the scene-logical semantics of the
    /// primitive bounds. It does not change to physical pixels when rendering
    /// into an isolated target. [`PrepareRegion`] provides explicit coordinate
    /// conversions and separates the valid viewport from the potentially padded
    /// backing extent.
    ///
    /// **Parameters**:
    ///
    /// * `pipeline`: the pipeline shared by all occurrences of this primitive
    ///   type
    /// * `device`: the WGPU device to use for allocating GPU resources
    /// * `queue`: the WGPU queue to use for uploading data to GPU resources
    /// * `target`: the primitive's scene-logical bounds together with the
    ///   target's valid viewport, full backing-texture extent, and mapping from
    ///   scene-logical coordinates to target-local physical pixels
    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &PrepareRegion,
    );

    /// Draws the [`Primitive`] in the given [`wgpu::RenderPass`].
    ///
    /// When possible, this should be implemented over [`render`](Self::render)
    /// since reusing the existing render pass should be considerably more
    /// efficient than issuing a new one.
    ///
    /// [`RenderRegion::configure`] has already been applied to `render_pass`.
    /// The region's two physical rectangles must not be confused:
    ///
    /// * The viewport maps normalized device coordinates to the complete,
    ///   uncut primitive bounds. A vertex shader which emits primitive-local
    ///   clip-space positions is therefore positioned and scaled by the
    ///   physical viewport. Active Iced transformations are already reflected
    ///   in this viewport and must not be applied a second time.
    /// * The scissor rectangle only discards fragments. It does not change the
    ///   coordinate mapping. It is the intersection of the primitive bounds,
    ///   active Iced clip, and valid target region, so it also prevents writes
    ///   to padding in a pooled backing texture.
    ///
    /// In particular, do not replace the viewport with the scissor rectangle:
    /// doing so would rescale the primitive whenever it is partially clipped.
    /// If the implementation changes either dynamic state, it must still
    /// restrict all color writes to [`RenderRegion::physical_scissor`].
    ///
    /// Iced transforms the primitive bounds, but it does not rewrite custom
    /// vertex data. Emitting primitive-local normalized device coordinates is
    /// the simplest way to inherit the transformed position and size through
    /// the configured viewport. An implementation which instead emits logical
    /// or target-local pixel coordinates can use the conversions supplied by the
    /// [`PrepareRegion`] received by [`Self::prepare`].
    ///
    /// If you have complex composition needs, then you can leverage
    /// [`render`](Self::render) by returning `false` here.
    ///
    /// Returning `false` means no draw commands were emitted. The renderer
    /// cannot roll back commands before calling [`Self::render`].
    ///
    /// **Parameters**:
    ///
    /// * `pipeline`: the shared pipeline previously initialized and prepared
    ///   for this primitive type
    /// * `region`: the logical primitive bounds and authoritative physical
    ///   viewport and scissor
    /// * `render_pass`: Iced's active color render pass, with its viewport set
    ///   to [`RenderRegion::physical_viewport`] and its scissor rectangle set to
    ///   [`RenderRegion::physical_scissor`]
    ///
    /// By default, this method does nothing and returns `false`.
    fn draw(
        &self,
        pipeline: &Self::Pipeline,
        region: &RenderRegion,
        render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let _ = (pipeline, region, render_pass);

        false
    }

    /// Renders the [`Primitive`], using the given [`wgpu::CommandEncoder`].
    ///
    /// This will only be called if [`draw`](Self::draw) returns `false`.
    ///
    /// A render pass created in this method defaults to the full backing
    /// attachment for both its viewport and scissor rectangle. Configure both
    /// states before drawing:
    ///
    /// ```rust
    /// # use iced_wgpu::primitive::RenderRegion;
    /// # use iced_wgpu::wgpu;
    /// # fn configure(
    /// #     render_pass: &mut wgpu::RenderPass<'_>,
    /// #     region: &RenderRegion,
    /// # ) {
    /// region.configure(render_pass);
    /// # }
    /// ```
    ///
    /// [`RenderRegion::configure`] keeps the viewport equal to the physical
    /// image of the complete, uncut primitive bounds, even when part of the
    /// primitive lies outside the target. The scissor then culls the invisible
    /// portion without stretching the remaining content.
    ///
    /// The supplied color target must be loaded and stored without clearing it.
    /// Auxiliary render attachments must match the full target extent and
    /// sample count. The full extent can also be obtained from
    /// `target_view.texture().size()` because Iced supplies a full base-mip
    /// target view. Blending and color writes must preserve Iced's
    /// premultiplied-alpha target contract.
    ///
    /// WGPU permits an uncut viewport to extend beyond the target, but still
    /// applies the device's viewport size and coordinate limits. Primitives
    /// larger than those limits need a custom projection or segmented render
    /// strategy.
    ///
    /// **Parameters**:
    ///
    /// * `pipeline`: the shared pipeline previously initialized and prepared
    ///   for this primitive type
    /// * `encoder`: the WGPU command encoder on which to create render passes or
    ///   encode other commands
    /// * `target_view`: the color attachment containing the existing Iced
    ///   output; it must be loaded, preserved, and updated without a clear
    /// * `region`: the logical primitive bounds and authoritative physical
    ///   viewport and scissor
    ///
    /// By default, it does nothing.
    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        region: &RenderRegion,
    ) {
        let _ = (pipeline, encoder, target_view, region);
    }
}

/// The pipeline of a graphics [`Primitive`].
pub trait Pipeline: Any + MaybeSend + MaybeSync {
    /// Creates the [`Pipeline`] of a [`Primitive`].
    ///
    /// This will only be called once per [`Primitive`] type and renderer, when
    /// the first primitive of that type is encountered.
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self
    where
        Self: Sized;

    /// Trims any cached data in the [`Pipeline`].
    ///
    /// This will normally be called at the end of a frame.
    fn trim(&mut self) {}
}

pub(crate) trait Stored: Debug + MaybeSend + MaybeSync + 'static {
    fn prepare(
        &self,
        storage: &mut Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        target: &PrepareRegion,
    );

    fn draw(
        &self,
        storage: &Storage,
        region: &RenderRegion,
        render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool;

    fn render(
        &self,
        storage: &Storage,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        region: &RenderRegion,
    );
}

#[derive(Debug)]
struct BlackBox<P: Primitive> {
    primitive: P,
}

impl<P: Primitive> Stored for BlackBox<P> {
    fn prepare(
        &self,
        storage: &mut Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        target: &PrepareRegion,
    ) {
        if !storage.has::<P>() {
            storage.store::<P, _>(P::Pipeline::new(device, queue, format));
        }

        let renderer = storage
            .get_mut::<P>()
            .expect("renderer should be initialized")
            .downcast_mut::<P::Pipeline>()
            .expect("renderer should have the proper type");

        self.primitive.prepare(renderer, device, queue, target);
    }

    fn draw(
        &self,
        storage: &Storage,
        region: &RenderRegion,
        render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let renderer = storage
            .get::<P>()
            .expect("renderer should be initialized")
            .downcast_ref::<P::Pipeline>()
            .expect("renderer should have the proper type");

        self.primitive.draw(renderer, region, render_pass)
    }

    fn render(
        &self,
        storage: &Storage,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        region: &RenderRegion,
    ) {
        let renderer = storage
            .get::<P>()
            .expect("renderer should be initialized")
            .downcast_ref::<P::Pipeline>()
            .expect("renderer should have the proper type");

        self.primitive
            .render(renderer, encoder, target_view, region);
    }
}

#[derive(Debug)]
/// An instance of a specific [`Primitive`].
pub struct Instance {
    /// The bounds of the [`Instance`].
    pub(crate) bounds: Rectangle,

    /// The [`Primitive`] to render.
    pub(crate) primitive: Box<dyn Stored>,
}

impl Instance {
    /// Creates a new [`Instance`] with the given [`Primitive`].
    ///
    /// **Parameters**:
    ///
    /// * `bounds`: the primitive bounds in Iced's current logical coordinate
    ///   system
    /// * `primitive`: the custom primitive to store in the instance
    pub fn new(bounds: Rectangle, primitive: impl Primitive) -> Self {
        Instance {
            bounds,
            primitive: Box::new(BlackBox { primitive }),
        }
    }
}

/// A renderer that can draw custom primitives.
pub trait Renderer: core::Renderer {
    /// Draws a custom primitive.
    ///
    /// **Parameters**:
    ///
    /// * `bounds`: the primitive bounds in Iced's current logical coordinate
    ///   system
    /// * `primitive`: the custom primitive to prepare and render in these bounds
    fn draw_primitive(&mut self, bounds: Rectangle, primitive: impl Primitive);
}

/// Stores custom, user-provided types.
#[derive(Default)]
pub struct Storage {
    pipelines: FxHashMap<TypeId, Box<dyn Pipeline>>,
}

impl Storage {
    /// Returns `true` if `Storage` contains a type `T`.
    ///
    /// **Type parameters**:
    ///
    /// * `T`: the primitive type whose pipeline entry should be checked
    pub fn has<T: 'static>(&self) -> bool {
        self.pipelines.contains_key(&TypeId::of::<T>())
    }

    /// Inserts a [`Pipeline`] into [`Storage`] under the type `T`.
    ///
    /// **Parameters**:
    ///
    /// * `pipeline`: the pipeline value to store under the type `T`, replacing
    ///   any existing value for the same type
    ///
    /// **Type parameters**:
    ///
    /// * `T`: the primitive type used as the storage key
    /// * `P`: the concrete pipeline type to store
    pub fn store<T: 'static, P: Pipeline>(&mut self, pipeline: P) {
        let _ = self.pipelines.insert(TypeId::of::<T>(), Box::new(pipeline));
    }

    /// Returns a reference to the data with type `T` if it exists in [`Storage`].
    ///
    /// **Type parameters**:
    ///
    /// * `T`: the primitive type whose pipeline entry should be returned
    pub fn get<T: 'static>(&self) -> Option<&dyn Any> {
        self.pipelines
            .get(&TypeId::of::<T>())
            .map(|pipeline| pipeline.as_ref() as &dyn Any)
    }

    /// Returns a mutable reference to the data with type `T` if it exists in [`Storage`].
    ///
    /// **Type parameters**:
    ///
    /// * `T`: the primitive type whose pipeline entry should be returned
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut dyn Any> {
        self.pipelines
            .get_mut(&TypeId::of::<T>())
            .map(|pipeline| pipeline.as_mut() as &mut dyn Any)
    }

    /// Trims the cache of all the pipelines in the [`Storage`].
    pub fn trim(&mut self) {
        for pipeline in self.pipelines.values_mut() {
            pipeline.trim();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Point;
    use crate::core::renderer;

    #[test]
    fn prepare_region_preserves_scene_logical_bounds_and_exposes_explicit_conversions() {
        let viewport = Viewport::with_physical_size(
            Size::new(80, 60),
            renderer::Scale {
                window: 2.0,
                application: 1.0,
            },
        );
        let bounds = Rectangle::new(Point::new(10.0, 25.0), Size::new(50.0, 30.0));
        let represented_bounds = Rectangle::new(Point::new(20.0, 30.0), Size::new(40.0, 30.0));
        let raster_transform = Transformation::translate(-40.0, -60.0) * Transformation::scale(2.0);
        let target = PrepareRegion::new(
            bounds,
            &viewport,
            represented_bounds,
            Size::new(128, 64),
            raster_transform,
        );

        assert_eq!(target.bounds(), &bounds);
        assert_eq!(target.represented_bounds(), represented_bounds);
        assert_eq!(target.viewport().physical_size(), Size::new(80, 60));
        assert_eq!(target.backing_extent(), Size::new(128, 64));
        assert_eq!(
            target.local_bounds(),
            Rectangle::new(Point::new(-10.0, -5.0), Size::new(50.0, 30.0))
        );
        assert_eq!(
            target.physical_bounds(),
            Rectangle::new(Point::new(-20.0, -10.0), Size::new(100.0, 60.0))
        );
        assert_eq!(target.logical_to_physical(), raster_transform);
        assert_eq!(
            target.logical_projection(),
            viewport.projection() * raster_transform
        );
    }

    #[test]
    fn render_region_keeps_logical_viewport_and_scissor_components_distinct() {
        let bounds = Rectangle::new(Point::new(10.0, 25.0), Size::new(50.0, 30.0));
        let physical_viewport = Rectangle::new(Point::new(-20.0, -10.0), Size::new(100.0, 60.0));
        let physical_scissor = Rectangle {
            x: 0,
            y: 0,
            width: 80,
            height: 50,
        };
        let region = RenderRegion::new(bounds, physical_viewport, physical_scissor);

        assert_eq!(region.bounds(), bounds);
        assert_eq!(region.physical_viewport(), physical_viewport);
        assert_eq!(region.physical_scissor(), physical_scissor);
    }
}
