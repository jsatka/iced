//! A [`wgpu`] renderer for [Iced].
//!
//! ![The native path of the Iced ecosystem](https://github.com/iced-rs/iced/blob/0525d76ff94e828b7b21634fa94a747022001c83/docs/graphs/native.png?raw=true)
//!
//! [`wgpu`] supports most modern graphics backends: Vulkan, Metal, DX11, and
//! DX12 (OpenGL and WebGL are still WIP). Additionally, it will support the
//! incoming [WebGPU API].
//!
//! Currently, `iced_wgpu` supports the following primitives:
//! - Text, which is rendered using [`glyphon`].
//! - Quads or rectangles, with rounded borders and a solid background color.
//! - Clip areas, useful to implement scrollables or hide overflowing content.
//! - Images and SVG, loaded from memory or the file system.
//! - Meshes of triangles, useful to draw geometry freely.
//!
//! [Iced]: https://github.com/iced-rs/iced
//! [`wgpu`]: https://github.com/gfx-rs/wgpu-rs
//! [WebGPU API]: https://gpuweb.github.io/gpuweb/
//! [`glyphon`]: https://github.com/grovesNL/glyphon
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/iced-rs/iced/9ab6923e943f784985e9ef9ca28b10278297225d/docs/logo.svg"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(missing_docs)]
pub mod isolated_layer;
pub mod layer;
pub mod primitive;
pub mod window;

#[doc(hidden)]
pub mod shader;

#[cfg(feature = "geometry")]
pub mod geometry;

mod buffer;
mod color;
mod engine;
mod quad;
mod text;
mod triangle;

#[cfg(any(feature = "image", feature = "svg"))]
#[path = "image/mod.rs"]
mod image;

#[cfg(not(any(feature = "image", feature = "svg")))]
#[path = "image/null.rs"]
mod image;

use buffer::Buffer;

use iced_debug as debug;
pub use iced_graphics as graphics;
pub use iced_graphics::core;

pub use wgpu;

pub use engine::Engine;
pub use layer::Layer;
pub use primitive::{PrepareRegion, Primitive, RenderRegion};

#[cfg(feature = "geometry")]
pub use geometry::Geometry;

use crate::core::renderer;
use crate::core::{Background, Color, Font, Pixels, Point, Rectangle, Size, Transformation};
use crate::graphics::mesh;
use crate::graphics::text::{Editor, Paragraph};
use crate::graphics::{Shell, Viewport};

/// A [`wgpu`] graphics renderer for [`iced`].
///
/// [`wgpu`]: https://github.com/gfx-rs/wgpu-rs
/// [`iced`]: https://github.com/iced-rs/iced
pub struct Renderer {
    engine: Engine,
    settings: renderer::Settings,

    layers: layer::Stack,
    recorder: isolated_layer::Recorder,
    isolated_layers: isolated_layer::State,
    scale: Option<renderer::Scale>,

    quad: quad::State,
    triangle: triangle::State,
    text: text::State,
    text_viewport: text::Viewport,
    segmented_text_viewport: Option<text::Viewport>,

    #[cfg(any(feature = "svg", feature = "image"))]
    image: image::State,

    // TODO: Centralize all the image feature handling
    #[cfg(any(feature = "svg", feature = "image"))]
    image_cache: std::cell::RefCell<image::Cache>,

    staging_belt: wgpu::util::StagingBelt,
}

impl Renderer {
    /// Returns diagnostics for the most recently rendered frame.
    pub fn isolated_layer_diagnostics(&self) -> isolated_layer::Diagnostics {
        self.isolated_layers.diagnostics
    }

    /// Returns the renderer-owned isolated-layer residency limits.
    pub fn isolated_layer_limits(&self) -> isolated_layer::Limits {
        self.isolated_layers.limits()
    }

    /// Replaces the renderer-owned isolated-layer residency limits.
    ///
    /// The new memory budget is enforced at the next rendered frame boundary.
    pub fn set_isolated_layer_limits(&mut self, limits: isolated_layer::Limits) {
        self.isolated_layers.set_limits(limits);
    }

    pub fn new(engine: Engine, settings: renderer::Settings) -> Self {
        Self {
            settings,
            layers: layer::Stack::new(),
            recorder: isolated_layer::Recorder::default(),
            isolated_layers: isolated_layer::State::default(),
            scale: None,

            quad: quad::State::new(),
            triangle: triangle::State::new(&engine.device, &engine.triangle_pipeline),
            text: text::State::new(),
            text_viewport: engine.text_pipeline.create_viewport(&engine.device),
            segmented_text_viewport: None,

            #[cfg(any(feature = "svg", feature = "image"))]
            image: image::State::new(),

            #[cfg(any(feature = "svg", feature = "image"))]
            image_cache: std::cell::RefCell::new(engine.create_image_cache()),

            // TODO: Resize belt smartly (?)
            // It would be great if the `StagingBelt` API exposed methods
            // for introspection to detect when a resize may be worth it.
            staging_belt: wgpu::util::StagingBelt::new(
                engine.device.clone(),
                buffer::MAX_WRITE_SIZE as u64,
            ),

            engine,
        }
    }

    /// Record commands that draw the current primitives to the target texture view.
    ///
    /// `target` must be the full, base-mip view of a single-layer, single-mip,
    /// sample-count-1 2D texture whose extent equals `viewport.physical_size()`
    /// and whose format matches this renderer. Arbitrary subresource views are
    /// not supported because WGPU does not expose their selected render extent.
    ///
    /// You must call [`finish`](Self::finish) and [`recall`](Self::recall) when submitting
    /// the resulting [`wgpu::CommandEncoder`].
    pub fn draw(
        &mut self,
        clear_color: Option<Color>,
        target: &wgpu::TextureView,
        viewport: &Viewport,
    ) -> wgpu::CommandEncoder {
        validate_root_target(target, viewport, self.engine.format);

        let mut encoder =
            self.engine
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("iced_wgpu encoder"),
                });

        let segmented = self.recorder.is_segmented();
        self.isolated_layers.begin_frame(segmented);

        if segmented {
            let mut sequence = self.recorder.take(&mut self.layers);
            self.draw_segmented(&mut encoder, target, clear_color, viewport, &mut sequence);
            self.release_prepared_sequence(&mut sequence);
            self.recorder.restore(sequence);
        } else {
            let context = isolated_layer::Context::root(viewport, self.engine.format);
            self.prepare(&mut encoder, viewport, &context);
            self.render(&mut encoder, target, clear_color, viewport, &context);
        }
        self.isolated_layers.finish_frame();

        self.quad.trim();
        self.triangle.trim();
        self.text.trim();

        // TODO: Provide window id (?)
        self.engine.trim();

        #[cfg(any(feature = "svg", feature = "image"))]
        {
            self.image.trim();
            self.image_cache.borrow_mut().trim();
        }

        encoder
    }

    pub fn present(
        &mut self,
        clear_color: Option<Color>,
        _format: wgpu::TextureFormat,
        frame: &wgpu::TextureView,
        viewport: &Viewport,
    ) -> wgpu::SubmissionIndex {
        let encoder = self.draw(clear_color, frame, viewport);

        self.staging_belt.finish();
        let submission = self.engine.queue.submit([encoder.finish()]);
        self.staging_belt.recall();
        submission
    }

    /// Renders the current surface to an offscreen buffer.
    ///
    /// Returns RGBA bytes of the texture data.
    pub fn screenshot(&mut self, viewport: &Viewport, background_color: Color) -> Vec<u8> {
        #[derive(Clone, Copy, Debug)]
        struct BufferDimensions {
            width: u32,
            height: u32,
            unpadded_bytes_per_row: usize,
            padded_bytes_per_row: usize,
        }

        impl BufferDimensions {
            fn new(size: Size<u32>) -> Self {
                let unpadded_bytes_per_row = size.width as usize * 4; //slice of buffer per row; always RGBA
                let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize; //256
                let padded_bytes_per_row_padding =
                    (alignment - unpadded_bytes_per_row % alignment) % alignment;
                let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

                Self {
                    width: size.width,
                    height: size.height,
                    unpadded_bytes_per_row,
                    padded_bytes_per_row,
                }
            }
        }

        let dimensions = BufferDimensions::new(viewport.physical_size());

        let texture_extent = wgpu::Extent3d {
            width: dimensions.width,
            height: dimensions.height,
            depth_or_array_layers: 1,
        };

        let texture = self.engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_wgpu.screenshot.source_texture"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.engine.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.draw(Some(background_color), &view, viewport);

        let texture = crate::color::convert(
            &self.engine.device,
            &mut encoder,
            texture,
            if graphics::color::GAMMA_CORRECTION {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
        );

        let output_buffer = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced_wgpu.screenshot.output_texture_buffer"),
            size: (dimensions.padded_bytes_per_row * dimensions.height as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dimensions.padded_bytes_per_row as u32),
                    rows_per_image: None,
                },
            },
            texture_extent,
        );

        self.staging_belt.finish();
        let index = self.engine.queue.submit([encoder.finish()]);
        self.staging_belt.recall();

        let slice = output_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});

        let _ = self.engine.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        });

        let mapped_buffer = slice.get_mapped_range();

        mapped_buffer
            .chunks(dimensions.padded_bytes_per_row)
            .fold(vec![], |mut acc, row| {
                acc.extend(&row[..dimensions.unpadded_bytes_per_row]);
                acc
            })
    }

    fn prepare(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        context: &isolated_layer::Context,
    ) {
        let scale_factor = viewport.scale_factor();

        self.text_viewport
            .update(&self.engine.queue, viewport.physical_size());

        let physical_bounds =
            Rectangle::<f32>::from(Rectangle::with_size(viewport.physical_size()));

        self.layers.merge();

        for layer in self.layers.iter() {
            let clip_bounds = layer.bounds * scale_factor;

            if physical_bounds
                .intersection(&clip_bounds)
                .and_then(Rectangle::snap)
                .is_none()
            {
                continue;
            }

            if !layer.quads.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Quad);

                self.quad.prepare(
                    &self.engine.quad_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &layer.quads,
                    viewport.projection(),
                    scale_factor,
                    Point::ORIGIN,
                );

                prepare_span.finish();
            }

            if !layer.triangles.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Triangle);

                self.triangle.prepare(
                    &self.engine.triangle_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &layer.triangles,
                    Transformation::scale(scale_factor),
                    viewport.physical_size(),
                );

                prepare_span.finish();
            }

            if !layer.primitives.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Shader);

                let mut primitive_storage = self
                    .engine
                    .primitive_storage
                    .write()
                    .expect("Write primitive storage");

                for instance in &layer.primitives {
                    let target = primitive::PrepareRegion::new(
                        instance.bounds,
                        viewport,
                        context.represented_bounds,
                        context.backing_extent(),
                        context.raster_transform,
                    );

                    instance.primitive.prepare(
                        &mut primitive_storage,
                        &self.engine.device,
                        &self.engine.queue,
                        context.format,
                        &target,
                    );
                }

                prepare_span.finish();
            }

            #[cfg(any(feature = "svg", feature = "image"))]
            if !layer.images.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Image);

                self.image.prepare(
                    &self.engine.image_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &mut self.image_cache.borrow_mut(),
                    &layer.images,
                    viewport.projection(),
                    Transformation::scale(scale_factor),
                );

                prepare_span.finish();
            }

            if !layer.text.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Text);

                self.text.prepare(
                    &self.engine.text_pipeline,
                    &self.engine.device,
                    &self.engine.queue,
                    &self.text_viewport,
                    encoder,
                    &layer.text,
                    layer.bounds,
                    Transformation::scale(scale_factor),
                );

                prepare_span.finish();
            }
        }
    }

    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::TextureView,
        clear_color: Option<Color>,
        viewport: &Viewport,
        context: &isolated_layer::Context,
    ) {
        use std::mem::ManuallyDrop;

        let mut render_pass =
            ManuallyDrop::new(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iced_wgpu render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match clear_color {
                            Some(background_color) => wgpu::LoadOp::Clear({
                                let [r, g, b, a] =
                                    graphics::color::pack(background_color).components();

                                wgpu::Color {
                                    r: f64::from(r * a),
                                    g: f64::from(g * a),
                                    b: f64::from(b * a),
                                    a: f64::from(a),
                                }
                            }),
                            None => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }));

        let mut quad_layer = 0;
        let mut mesh_layer = 0;
        let mut text_layer = 0;

        #[cfg(any(feature = "svg", feature = "image"))]
        let mut image_layer = 0;

        let scale_factor = viewport.scale_factor();
        let physical_bounds =
            Rectangle::<f32>::from(Rectangle::with_size(viewport.physical_size()));

        let raster_transform = context.raster_transform;

        for layer in self.layers.iter() {
            let Some(physical_bounds) =
                physical_bounds.intersection(&(layer.bounds * scale_factor))
            else {
                continue;
            };

            let Some(scissor_rect) = physical_bounds.snap() else {
                continue;
            };

            if !layer.quads.is_empty() {
                let render_span = debug::render(debug::Primitive::Quad);
                self.quad.render(
                    &self.engine.quad_pipeline,
                    quad_layer,
                    scissor_rect,
                    &layer.quads,
                    &mut render_pass,
                );
                render_span.finish();

                quad_layer += 1;
            }

            if !layer.triangles.is_empty() {
                let _ = ManuallyDrop::into_inner(render_pass);

                let render_span = debug::render(debug::Primitive::Triangle);
                mesh_layer += self.triangle.render(
                    &self.engine.triangle_pipeline,
                    encoder,
                    frame,
                    viewport.physical_size(),
                    mesh_layer,
                    &layer.triangles,
                    physical_bounds,
                    raster_transform,
                );
                render_span.finish();

                render_pass =
                    ManuallyDrop::new(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("iced_wgpu render pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: frame,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    }));
            }

            if !layer.primitives.is_empty() {
                let render_span = debug::render(debug::Primitive::Shader);

                let primitive_storage = self
                    .engine
                    .primitive_storage
                    .read()
                    .expect("Read primitive storage");

                for instance in &layer.primitives {
                    if let Some(region) =
                        primitive_region(context, instance.bounds, physical_bounds)
                    {
                        region.configure(&mut render_pass);

                        let drawn =
                            instance
                                .primitive
                                .draw(&primitive_storage, &region, &mut render_pass);

                        if !drawn {
                            let _ = ManuallyDrop::into_inner(render_pass);
                            instance
                                .primitive
                                .render(&primitive_storage, encoder, frame, &region);
                            render_pass = ManuallyDrop::new(begin_load_pass(
                                encoder,
                                frame,
                                viewport.physical_size(),
                            ));
                        }
                    }
                }

                render_pass.set_viewport(
                    0.0,
                    0.0,
                    viewport.physical_width() as f32,
                    viewport.physical_height() as f32,
                    0.0,
                    1.0,
                );

                render_pass.set_scissor_rect(
                    0,
                    0,
                    viewport.physical_width(),
                    viewport.physical_height(),
                );

                render_span.finish();
            }

            #[cfg(any(feature = "svg", feature = "image"))]
            if !layer.images.is_empty() {
                let render_span = debug::render(debug::Primitive::Image);
                self.image.render(
                    &self.engine.image_pipeline,
                    image_layer,
                    scissor_rect,
                    &mut render_pass,
                );
                render_span.finish();

                image_layer += 1;
            }

            if !layer.text.is_empty() {
                let render_span = debug::render(debug::Primitive::Text);
                text_layer += self.text.render(
                    &self.engine.text_pipeline,
                    &self.text_viewport,
                    text_layer,
                    &layer.text,
                    scissor_rect,
                    &mut render_pass,
                );
                render_span.finish();
            }
        }

        let _ = ManuallyDrop::into_inner(render_pass);

        debug::layers_rendered(|| {
            self.layers
                .iter()
                .filter(|layer| {
                    !layer.is_empty()
                        && physical_bounds
                            .intersection(&(layer.bounds * scale_factor))
                            .is_some_and(|viewport| viewport.snap().is_some())
                })
                .count()
        });
    }

    fn draw_segmented(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::TextureView,
        clear_color: Option<Color>,
        viewport: &Viewport,
        sequence: &mut isolated_layer::Sequence,
    ) {
        let needs_root_intermediate = sequence.needs_backdrop();
        self.isolated_layers.diagnostics.root_intermediate = needs_root_intermediate;

        let mut root_context = isolated_layer::Context::root(viewport, self.engine.format);

        if needs_root_intermediate {
            let (mut root, hit) = self.isolated_layers.pool.lease(
                &self.engine.device,
                &self.engine.text_pipeline,
                self.engine.format,
                viewport.physical_size(),
                self.isolated_layers.frame,
            );
            self.record_pool_result(hit);
            root_context.set_backing_extent(root.extent);
            root.text_viewport
                .update(&self.engine.queue, viewport.physical_size());

            self.prepare_sequence(encoder, sequence, &root_context, &mut root.text_viewport, 0);
            clear_view(
                encoder,
                &root.view,
                wgpu::LoadOp::Clear(clear_color.map_or(wgpu::Color::TRANSPARENT, packed_color)),
            );
            self.render_sequence(
                encoder,
                sequence,
                &root.view,
                Some(&root.texture),
                &root_context,
                &root.text_viewport,
            );

            let composite = self
                .engine
                .composite_storage
                .write()
                .expect("Write isolated-layer composite storage")
                .prepare(
                    &self.engine.device,
                    self.engine.format,
                    &root,
                    &root_context,
                    core::isolated_layer::Composite::default(),
                );
            let destination = Rectangle::with_size(viewport.physical_size());
            isolated_layer::render_composite(
                encoder,
                frame,
                viewport.physical_size(),
                root_context.placement,
                destination,
                if clear_color.is_some() {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                },
                &composite,
            );
            drop(composite);
            self.isolated_layers
                .pool
                .release(root, self.isolated_layers.frame);
        } else {
            let mut root_text_viewport = self.segmented_text_viewport.take().unwrap_or_else(|| {
                self.engine
                    .text_pipeline
                    .create_viewport(&self.engine.device)
            });
            root_text_viewport.update(&self.engine.queue, viewport.physical_size());

            self.prepare_sequence(encoder, sequence, &root_context, &mut root_text_viewport, 0);
            clear_view(
                encoder,
                frame,
                clear_color.map_or(wgpu::LoadOp::Load, |color| {
                    wgpu::LoadOp::Clear(packed_color(color))
                }),
            );
            self.render_sequence(
                encoder,
                sequence,
                frame,
                None,
                &root_context,
                &root_text_viewport,
            );
            self.segmented_text_viewport = Some(root_text_viewport);
        }
    }

    fn prepare_sequence(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        sequence: &mut isolated_layer::Sequence,
        parent: &isolated_layer::Context,
        text_viewport: &mut text::Viewport,
        depth: usize,
    ) {
        self.isolated_layers.diagnostics.max_depth =
            self.isolated_layers.diagnostics.max_depth.max(depth);

        for node in &mut sequence.0 {
            match node {
                isolated_layer::Node::Leaf(leaf) => {
                    self.prepare_leaf(encoder, leaf, parent, text_viewport);
                }
                isolated_layer::Node::Layer(node) => {
                    let grid = capture_grid(
                        node.layer.output_cache_request.is_some(),
                        node.layer.composite.positioning(),
                    );
                    let Some(mut context) = bounded_context(
                        node.layer.bounds,
                        parent,
                        grid,
                        self.engine.device.limits().max_texture_dimension_2d,
                    ) else {
                        continue;
                    };
                    context.set_source_geometry(node.logical_surface_size, node.layer.bounds);

                    let (retained, output_valid, output_lease) =
                        if node.layer.output_cache_request.is_some() {
                            self.lease_output(
                                node.layer.output_cache_request.as_ref(),
                                isolated_layer::EffectStack::new().input_evidence(),
                                &context,
                                node.layer.bounds,
                                node.layer.content_depends_on_translation,
                            )
                        } else {
                            (None, false, None)
                        };
                    let output =
                        retained.unwrap_or_else(|| self.lease_target(context.physical_viewport()));
                    let composite = self
                        .engine
                        .composite_storage
                        .write()
                        .expect("Write isolated-layer composite storage")
                        .prepare(
                            &self.engine.device,
                            self.engine.format,
                            &output,
                            &context,
                            node.layer.composite,
                        );

                    let mut targets = vec![output];
                    targets[0]
                        .text_viewport
                        .update(&self.engine.queue, context.physical_viewport());
                    node.prepared = Some(isolated_layer::PreparedIsolatedLayer {
                        context: context.clone(),
                        targets,
                        composite,
                        output_lease,
                        output_valid,
                    });
                    self.isolated_layers.diagnostics.nodes += 1;

                    let prepared = node.prepared.as_mut().expect("prepared node");
                    if !prepared.output_valid {
                        self.prepare_sequence(
                            encoder,
                            &mut node.content,
                            &context,
                            &mut prepared.targets[0].text_viewport,
                            depth + 1,
                        );
                    }
                }
                isolated_layer::Node::Effect(node) => {
                    let grid = capture_grid(
                        node.layer.output_cache_request.is_some(),
                        node.layer.composite.positioning(),
                    );
                    let Some(mut context) = bounded_context(
                        node.layer.bounds,
                        parent,
                        grid,
                        self.engine.device.limits().max_texture_dimension_2d,
                    ) else {
                        continue;
                    };
                    context
                        .set_source_geometry(node.logical_surface_size, node.layer.content_bounds);

                    // Snapshot cache evidence before asking the same effect values to build the
                    // executable plan. If an interior-mutable implementation changes its
                    // requirements between these observations, store-time recollection rejects
                    // the candidate instead of publishing pixels produced by a different plan.
                    let input_evidence = node
                        .layer
                        .output_cache_request
                        .is_some()
                        .then(|| node.effects.input_evidence());
                    let pass_plan = isolated_layer::plan_effect_passes(&node.effects);
                    let pass_count = pass_plan.passes.len();
                    let position_sensitive = node.layer.content_depends_on_translation
                        || !node.effects.is_translation_invariant();
                    let (retained, output_valid, output_lease) = if pass_plan.backdrop.is_some()
                        && node.layer.output_cache_request.is_some()
                    {
                        self.isolated_layers.diagnostics.output_cache_misses = self
                            .isolated_layers
                            .diagnostics
                            .output_cache_misses
                            .saturating_add(1);
                        self.isolated_layers
                            .diagnostics
                            .output_cache_bypass_backdrop = self
                            .isolated_layers
                            .diagnostics
                            .output_cache_bypass_backdrop
                            .saturating_add(1);
                        (None, false, None)
                    } else if node.layer.output_cache_request.is_some() {
                        self.lease_output(
                            node.layer.output_cache_request.as_ref(),
                            input_evidence.expect("cache evidence was requested"),
                            &context,
                            node.layer.content_bounds,
                            position_sensitive,
                        )
                    } else {
                        (None, false, None)
                    };
                    let mut targets = vec![
                        retained.unwrap_or_else(|| self.lease_target(context.physical_viewport())),
                    ];
                    let mut backdrop = None;
                    let pass_context =
                        isolated_layer::effect_context(&context, node.layer.content_bounds);
                    let mut prepared_passes = Vec::new();
                    if !output_valid {
                        if let Some(index) = pass_plan.backdrop {
                            debug_assert_eq!(index, targets.len());
                            targets.push(self.lease_target(context.physical_viewport()));
                            backdrop = Some(index);
                        }
                        prepared_passes.reserve(pass_count);
                        while targets.len() < pass_plan.target_count {
                            targets.push(self.lease_target(context.physical_viewport()));
                        }

                        let mut storage = self
                            .engine
                            .layer_effect_storage
                            .write()
                            .expect("Write isolated-layer effect storage");

                        for planned in &pass_plan.passes {
                            let effect = &node.effects[planned.effect];
                            let prepared = effect.stored().prepare_pass(
                                &mut storage,
                                &self.engine.device,
                                &self.engine.queue,
                                self.engine.format,
                                planned.pass,
                                &pass_context,
                                isolated_layer::TextureViews {
                                    stage_input: &targets[planned.stage_input].view,
                                    backdrop: if planned.uses_backdrop {
                                        backdrop.map(|index| &targets[index].view)
                                    } else {
                                        None
                                    },
                                    previous: &targets[planned.previous].view,
                                    output: &targets[planned.output].view,
                                },
                            );
                            prepared_passes.push(isolated_layer::PreparedEffectPass {
                                effect: planned.effect,
                                pass: planned.pass,
                                stage_input: planned.stage_input,
                                previous: planned.previous,
                                output: planned.output,
                                uses_backdrop: planned.uses_backdrop,
                                writes_every_pixel: planned.writes_every_pixel,
                                prepared,
                            });
                        }
                    }
                    let output = if output_valid { 0 } else { pass_plan.output };
                    let composite = self
                        .engine
                        .composite_storage
                        .write()
                        .expect("Write isolated-layer composite storage")
                        .prepare(
                            &self.engine.device,
                            self.engine.format,
                            &targets[output],
                            &context,
                            node.layer.composite,
                        );

                    targets[0]
                        .text_viewport
                        .update(&self.engine.queue, context.physical_viewport());
                    node.prepared = Some(isolated_layer::PreparedLayerEffect {
                        context: context.clone(),
                        targets,
                        backdrop,
                        composite,
                        passes: prepared_passes,
                        output,
                        output_lease,
                        output_valid,
                    });
                    self.isolated_layers.diagnostics.nodes += 1;
                    if !output_valid {
                        self.isolated_layers.diagnostics.isolated_layer_passes += pass_count;
                    }

                    let prepared = node
                        .prepared
                        .as_mut()
                        .expect("prepared isolated-layer effect");
                    if !prepared.output_valid {
                        self.prepare_sequence(
                            encoder,
                            &mut node.content,
                            &context,
                            &mut prepared.targets[0].text_viewport,
                            depth + 1,
                        );
                    }
                }
            }
        }
    }

    fn prepare_leaf(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        leaf: &mut isolated_layer::Leaf,
        context: &isolated_layer::Context,
        text_viewport: &text::Viewport,
    ) {
        leaf.stack.merge();
        leaf.prepared.clear();

        let target_viewport = context.viewport();

        for layer in leaf.stack.iter() {
            let Some(physical_bounds) = context.local_bounds(layer.bounds) else {
                leaf.prepared.push(None);
                continue;
            };
            let Some(scissor) = physical_bounds.snap() else {
                leaf.prepared.push(None);
                continue;
            };

            let quad = if layer.quads.is_empty() {
                None
            } else {
                let index = self.quad.prepared_layer_count();
                let translation = context.raster_transform.translation();
                self.quad.prepare(
                    &self.engine.quad_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &layer.quads,
                    target_viewport.projection(),
                    context.scale_factor(),
                    Point::new(
                        -translation.x / context.scale_factor(),
                        -translation.y / context.scale_factor(),
                    ),
                );
                Some(index)
            };

            let triangle_start = self.triangle.prepared_layer_count();
            if !layer.triangles.is_empty() {
                self.triangle.prepare(
                    &self.engine.triangle_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &layer.triangles,
                    context.raster_transform,
                    context.physical_viewport(),
                );
            }
            let triangle_end = self.triangle.prepared_layer_count();

            if !layer.primitives.is_empty() {
                let mut primitive_storage = self
                    .engine
                    .primitive_storage
                    .write()
                    .expect("Write primitive storage");

                for instance in &layer.primitives {
                    let target = primitive::PrepareRegion::new(
                        instance.bounds,
                        &target_viewport,
                        context.represented_bounds,
                        context.backing_extent(),
                        context.raster_transform,
                    );

                    instance.primitive.prepare(
                        &mut primitive_storage,
                        &self.engine.device,
                        &self.engine.queue,
                        context.format,
                        &target,
                    );
                }
            }

            #[cfg(any(feature = "svg", feature = "image"))]
            let image = if layer.images.is_empty() {
                None
            } else {
                let index = self.image.prepared_layer_count();
                self.image.prepare(
                    &self.engine.image_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &mut self.image_cache.borrow_mut(),
                    &layer.images,
                    target_viewport.projection(),
                    context.raster_transform,
                );
                Some(index)
            };

            let text_start = self.text.prepared_layer_count();
            if !layer.text.is_empty() {
                self.text.prepare(
                    &self.engine.text_pipeline,
                    &self.engine.device,
                    &self.engine.queue,
                    text_viewport,
                    encoder,
                    &layer.text,
                    layer.bounds,
                    context.raster_transform,
                );
            }
            let text_end = self.text.prepared_layer_count();

            leaf.prepared.push(Some(isolated_layer::PreparedLayer {
                scissor,
                physical_bounds,
                quad,
                triangle: triangle_start..triangle_end,
                #[cfg(any(feature = "svg", feature = "image"))]
                image,
                text: text_start..text_end,
            }));
        }
    }

    fn render_sequence(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        sequence: &mut isolated_layer::Sequence,
        target_view: &wgpu::TextureView,
        target_texture: Option<&wgpu::Texture>,
        context: &isolated_layer::Context,
        text_viewport: &text::Viewport,
    ) {
        for node in &mut sequence.0 {
            match node {
                isolated_layer::Node::Leaf(leaf) => {
                    self.render_leaf(encoder, leaf, target_view, context, text_viewport);
                }
                isolated_layer::Node::Layer(node) => {
                    let Some(prepared) = node.prepared.as_ref() else {
                        continue;
                    };

                    if !prepared.output_valid {
                        clear_view(
                            encoder,
                            &prepared.targets[0].view,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                        self.render_sequence(
                            encoder,
                            &mut node.content,
                            &prepared.targets[0].view,
                            Some(&prepared.targets[0].texture),
                            &prepared.context,
                            &prepared.targets[0].text_viewport,
                        );
                    }

                    let Some(clip) = context.local_scissor(node.layer.clip) else {
                        continue;
                    };
                    isolated_layer::render_composite(
                        encoder,
                        target_view,
                        context.physical_viewport(),
                        prepared.context.placement,
                        clip,
                        wgpu::LoadOp::Load,
                        &prepared.composite,
                    );
                }
                isolated_layer::Node::Effect(node) => {
                    let Some(prepared) = node.prepared.as_ref() else {
                        continue;
                    };

                    if let Some(backdrop) = prepared.backdrop {
                        if node.layer.composite.positioning()
                            == core::isolated_layer::CompositePositioning::Subpixel
                        {
                            let backdrop_blit = self
                                .engine
                                .composite_storage
                                .write()
                                .expect("Write isolated-layer composite storage")
                                .prepare_backdrop(
                                    &self.engine.device,
                                    self.engine.format,
                                    target_view,
                                    context,
                                    &prepared.context,
                                );
                            isolated_layer::render_backdrop(
                                encoder,
                                &prepared.targets[backdrop].view,
                                prepared.context.physical_viewport(),
                                &backdrop_blit,
                            );
                        } else {
                            let parent_texture = target_texture.expect(
                                "backdrop planning must provide a sampleable parent target",
                            );
                            let destination = prepared.context.placement.snapped();
                            encoder.copy_texture_to_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: parent_texture,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d {
                                        x: destination.x,
                                        y: destination.y,
                                        z: 0,
                                    },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                prepared.targets[backdrop].texture.as_image_copy(),
                                wgpu::Extent3d {
                                    width: prepared.context.physical_viewport().width,
                                    height: prepared.context.physical_viewport().height,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                    }

                    if !prepared.output_valid {
                        clear_view(
                            encoder,
                            &prepared.targets[0].view,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                        self.render_sequence(
                            encoder,
                            &mut node.content,
                            &prepared.targets[0].view,
                            Some(&prepared.targets[0].texture),
                            &prepared.context,
                            &prepared.targets[0].text_viewport,
                        );
                    }

                    let pass_context = isolated_layer::effect_context(
                        &prepared.context,
                        node.layer.content_bounds,
                    );
                    if !prepared.passes.is_empty() {
                        let storage = self
                            .engine
                            .layer_effect_storage
                            .read()
                            .expect("Read isolated-layer effect storage");
                        for pass in &prepared.passes {
                            if !pass.writes_every_pixel {
                                clear_view(
                                    encoder,
                                    &prepared.targets[pass.output].view,
                                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                );
                            }

                            node.effects[pass.effect].stored().encode_pass(
                                &storage,
                                &self.engine.device,
                                &self.engine.queue,
                                self.engine.format,
                                pass.prepared.as_ref(),
                                encoder,
                                pass.pass,
                                &pass_context,
                                isolated_layer::TextureViews {
                                    stage_input: &prepared.targets[pass.stage_input].view,
                                    backdrop: if pass.uses_backdrop {
                                        prepared.backdrop.map(|index| &prepared.targets[index].view)
                                    } else {
                                        None
                                    },
                                    previous: &prepared.targets[pass.previous].view,
                                    output: &prepared.targets[pass.output].view,
                                },
                            );
                        }
                    }

                    let Some(clip) = context.local_scissor(node.layer.clip) else {
                        continue;
                    };
                    isolated_layer::render_composite(
                        encoder,
                        target_view,
                        context.physical_viewport(),
                        prepared.context.placement,
                        clip,
                        wgpu::LoadOp::Load,
                        &prepared.composite,
                    );
                }
            }
        }
    }

    fn render_leaf(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        leaf: &isolated_layer::Leaf,
        target: &wgpu::TextureView,
        context: &isolated_layer::Context,
        text_viewport: &text::Viewport,
    ) {
        use std::mem::ManuallyDrop;

        let mut render_pass = ManuallyDrop::new(begin_load_pass(
            encoder,
            target,
            context.physical_viewport(),
        ));

        for (layer, prepared) in leaf.stack.iter().zip(&leaf.prepared) {
            let Some(prepared) = prepared else {
                continue;
            };

            if let Some(index) = prepared.quad {
                self.quad.render(
                    &self.engine.quad_pipeline,
                    index,
                    prepared.scissor,
                    &layer.quads,
                    &mut render_pass,
                );
            }

            if !prepared.triangle.is_empty() {
                let _ = ManuallyDrop::into_inner(render_pass);
                let _ = self.triangle.render(
                    &self.engine.triangle_pipeline,
                    encoder,
                    target,
                    context.physical_viewport(),
                    prepared.triangle.start,
                    &layer.triangles,
                    prepared.physical_bounds,
                    context.raster_transform,
                );
                render_pass = ManuallyDrop::new(begin_load_pass(
                    encoder,
                    target,
                    context.physical_viewport(),
                ));
            }

            if !layer.primitives.is_empty() {
                let primitive_storage = self
                    .engine
                    .primitive_storage
                    .read()
                    .expect("Read primitive storage");
                for instance in &layer.primitives {
                    if let Some(region) =
                        primitive_region(context, instance.bounds, prepared.physical_bounds)
                    {
                        region.configure(&mut render_pass);

                        if !instance
                            .primitive
                            .draw(&primitive_storage, &region, &mut render_pass)
                        {
                            let _ = ManuallyDrop::into_inner(render_pass);
                            instance
                                .primitive
                                .render(&primitive_storage, encoder, target, &region);
                            render_pass = ManuallyDrop::new(begin_load_pass(
                                encoder,
                                target,
                                context.physical_viewport(),
                            ));
                        }
                    }
                }

                render_pass.set_viewport(
                    0.0,
                    0.0,
                    context.physical_viewport().width as f32,
                    context.physical_viewport().height as f32,
                    0.0,
                    1.0,
                );
                render_pass.set_scissor_rect(
                    0,
                    0,
                    context.physical_viewport().width,
                    context.physical_viewport().height,
                );
            }

            #[cfg(any(feature = "svg", feature = "image"))]
            if let Some(index) = prepared.image {
                self.image.render(
                    &self.engine.image_pipeline,
                    index,
                    prepared.scissor,
                    &mut render_pass,
                );
            }

            if !prepared.text.is_empty() {
                let _ = self.text.render(
                    &self.engine.text_pipeline,
                    text_viewport,
                    prepared.text.start,
                    &layer.text,
                    prepared.scissor,
                    &mut render_pass,
                );
            }
        }

        let _ = ManuallyDrop::into_inner(render_pass);
    }

    fn lease_target(&mut self, requested_size: Size<u32>) -> isolated_layer::Target {
        let (target, hit) = self.isolated_layers.pool.lease(
            &self.engine.device,
            &self.engine.text_pipeline,
            self.engine.format,
            requested_size,
            self.isolated_layers.frame,
        );
        self.record_pool_result(hit);
        target
    }

    fn lease_output(
        &mut self,
        cache_request: Option<&core::isolated_layer::CacheRequest>,
        evidence: isolated_layer::LayerInputEvidence,
        context: &isolated_layer::Context,
        content_bounds: Rectangle,
        position_sensitive: bool,
    ) -> (
        Option<isolated_layer::Target>,
        bool,
        Option<(
            core::isolated_layer::CacheRequest,
            isolated_layer::OutputKey,
            isolated_layer::LeaseTicket,
        )>,
    ) {
        let Some(cache_request) = cache_request else {
            return (None, false, None);
        };

        // Widget drawing has already completed by this late lookup. Resampling here lets a
        // custom child widget mark its shared content handle while it records this frame.
        let cache_request = cache_request.resnapshot();
        let key = isolated_layer::OutputKey::new(
            &cache_request,
            evidence,
            context,
            content_bounds,
            position_sensitive,
            self.engine.device_epoch,
        );
        let lease = self.isolated_layers.registry.lease_output(
            &cache_request,
            key.clone(),
            self.isolated_layers.frame,
        );

        if lease.priority_conflict {
            self.isolated_layers
                .diagnostics
                .residency_priority_conflicts = self
                .isolated_layers
                .diagnostics
                .residency_priority_conflicts
                .saturating_add(1);
        }
        if let Some(miss) = lease.miss {
            self.isolated_layers.record_output_miss(miss);
        }

        if !lease.cacheable {
            return (None, false, None);
        }

        debug_assert_eq!(
            lease.target.as_ref().map(|target| target.extent),
            lease.valid.then(|| context.backing_extent()),
            "a retained output hit must have the keyed target extent",
        );

        if lease.valid {
            self.isolated_layers.diagnostics.output_cache_hits = self
                .isolated_layers
                .diagnostics
                .output_cache_hits
                .saturating_add(1);
        }

        (
            lease.target,
            lease.valid,
            Some((
                cache_request,
                key,
                lease.ticket.expect("cacheable output lease ticket"),
            )),
        )
    }

    fn record_pool_result(&mut self, hit: bool) {
        if hit {
            self.isolated_layers.diagnostics.pool_hits += 1;
        } else {
            self.isolated_layers.diagnostics.allocations += 1;
        }
    }

    fn release_prepared_sequence(&mut self, sequence: &mut isolated_layer::Sequence) {
        for node in &mut sequence.0 {
            if let isolated_layer::Node::Leaf(leaf) = node {
                leaf.prepared.clear();
            } else if let isolated_layer::Node::Layer(node) = node {
                self.release_prepared_sequence(&mut node.content);

                if let Some(prepared) = node.prepared.take() {
                    let isolated_layer::PreparedIsolatedLayer {
                        targets,
                        composite,
                        output_lease,
                        context,
                        ..
                    } = prepared;
                    drop(composite);

                    let mut targets: Vec<_> = targets.into_iter().map(Some).collect();
                    if let Some((cache_request, _recorded_key, ticket)) = output_lease {
                        let fresh_key = isolated_layer::OutputKey::new(
                            &cache_request,
                            isolated_layer::EffectStack::new().input_evidence(),
                            &context,
                            node.layer.bounds,
                            node.layer.content_depends_on_translation,
                            self.engine.device_epoch,
                        );
                        let output = targets[0].take().expect("leased output target");
                        let outcome = self.isolated_layers.registry.store_output(
                            ticket,
                            &cache_request,
                            fresh_key,
                            self.isolated_layers.frame,
                            output,
                        );
                        let _ = self.isolated_layers.record_store(outcome);
                    }
                    for target in targets.into_iter().flatten() {
                        self.isolated_layers
                            .pool
                            .release(target, self.isolated_layers.frame);
                    }
                }
            } else if let isolated_layer::Node::Effect(node) = node {
                self.release_prepared_sequence(&mut node.content);

                if let Some(prepared) = node.prepared.take() {
                    let isolated_layer::PreparedLayerEffect {
                        targets,
                        passes,
                        composite,
                        output_lease,
                        output,
                        context,
                        ..
                    } = prepared;
                    drop(passes);
                    drop(composite);

                    let mut targets: Vec<_> = targets.into_iter().map(Some).collect();
                    if let Some((cache_request, _recorded_key, ticket)) = output_lease {
                        let fresh_key = isolated_layer::OutputKey::new(
                            &cache_request,
                            node.effects.recollect_input_evidence(),
                            &context,
                            node.layer.content_bounds,
                            node.layer.content_depends_on_translation
                                || !node.effects.is_translation_invariant(),
                            self.engine.device_epoch,
                        );
                        let retained = targets[output].take().expect("leased output target");
                        let outcome = self.isolated_layers.registry.store_output(
                            ticket,
                            &cache_request,
                            fresh_key,
                            self.isolated_layers.frame,
                            retained,
                        );
                        let _ = self.isolated_layers.record_store(outcome);
                    }
                    for target in targets.into_iter().flatten() {
                        self.isolated_layers
                            .pool
                            .release(target, self.isolated_layers.frame);
                    }
                }
            }
        }
    }

    /// Prepares currently mapped buffers for use in a submission.
    ///
    /// Usually, this method is only needed if you are calling [`Renderer::draw`] directly,
    /// instead of relying on [`Renderer::present`].
    ///
    /// You must call this method _before_ submitting the resulting [`wgpu::CommandEncoder`]
    /// of [`Renderer::draw`] to a [`wgpu::Queue`].
    pub fn finish(&mut self) {
        self.staging_belt.finish();
    }

    /// Recalls all of the closed buffers back to be reused.
    ///
    /// Usually, this method is only needed if you are calling [`Renderer::draw`] directly,
    /// instead of relying on [`Renderer::present`] to a [`wgpu::Queue`].
    ///
    /// You must call this method _after_ submitting the resulting [`wgpu::CommandEncoder`]
    /// of [`Renderer::draw`] to a [`wgpu::Queue`].
    pub fn recall(&mut self) {
        self.staging_belt.recall();
    }
}

fn validate_root_target(
    target: &wgpu::TextureView,
    viewport: &Viewport,
    format: wgpu::TextureFormat,
) {
    let texture = target.texture();
    let extent = texture.size();
    let physical_size = viewport.physical_size();

    assert_eq!(
        texture.dimension(),
        wgpu::TextureDimension::D2,
        "iced_wgpu root target must be a 2D texture"
    );
    assert_eq!(
        extent.depth_or_array_layers, 1,
        "iced_wgpu root target must have one array layer"
    );
    assert_eq!(
        texture.mip_level_count(),
        1,
        "iced_wgpu root target must have one mip level"
    );
    assert_eq!(
        texture.sample_count(),
        1,
        "iced_wgpu root target must have sample count 1"
    );
    assert_eq!(
        texture.format(),
        format,
        "iced_wgpu root target format must match its renderer"
    );
    assert_eq!(
        Size::new(extent.width, extent.height),
        physical_size,
        "iced_wgpu root target extent must match the viewport"
    );
}

fn primitive_region(
    context: &isolated_layer::Context,
    bounds: Rectangle,
    active_clip: Rectangle,
) -> Option<primitive::RenderRegion> {
    let physical_viewport = context.raster_bounds(bounds);
    let valid_bounds = Rectangle::<f32>::from(Rectangle::with_size(context.physical_viewport()));
    let physical_scissor = physical_viewport
        .intersection(&active_clip)?
        .intersection(&valid_bounds)?
        .snap()?;

    debug_assert!(
        physical_scissor.x.saturating_add(physical_scissor.width)
            <= context.physical_viewport().width
            && physical_scissor.y.saturating_add(physical_scissor.height)
                <= context.physical_viewport().height,
        "custom primitive clip bounds must lie inside the valid target viewport"
    );

    Some(primitive::RenderRegion::new(
        bounds,
        physical_viewport,
        physical_scissor,
    ))
}

fn begin_load_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    target: &'a wgpu::TextureView,
    viewport: Size<u32>,
) -> wgpu::RenderPass<'a> {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("iced_wgpu load render pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_viewport(
        0.0,
        0.0,
        viewport.width as f32,
        viewport.height as f32,
        0.0,
        1.0,
    );
    render_pass.set_scissor_rect(0, 0, viewport.width, viewport.height);
    render_pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root_context(physical_size: Size<u32>, scale_factor: f32) -> isolated_layer::Context {
        let viewport = Viewport::with_physical_size(
            physical_size,
            renderer::Scale {
                window: scale_factor,
                application: 1.0,
            },
        );

        isolated_layer::Context::root(&viewport, wgpu::TextureFormat::Rgba8Unorm)
    }

    #[test]
    fn primitive_region_uses_physical_bounds_and_clips_to_active_region() {
        let context = root_context(Size::new(200, 100), 2.0);

        let region = primitive_region(
            &context,
            Rectangle::new(Point::new(10.0, 5.0), Size::new(20.0, 10.0)),
            Rectangle::new(Point::new(25.0, 12.0), Size::new(20.0, 10.0)),
        );

        assert_eq!(
            region,
            Some(primitive::RenderRegion::new(
                Rectangle::new(Point::new(10.0, 5.0), Size::new(20.0, 10.0)),
                Rectangle::new(Point::new(20.0, 10.0), Size::new(40.0, 20.0)),
                Rectangle {
                    x: 25,
                    y: 12,
                    width: 20,
                    height: 10,
                },
            ))
        );
    }

    #[test]
    fn primitive_region_preserves_uncut_isolated_bounds_and_excludes_padding() {
        let root = root_context(Size::new(200, 200), 2.0);
        let mut context = isolated_layer::Context::bounded_with_grid(
            Rectangle::new(Point::new(20.0, 30.0), Size::new(40.0, 30.0)),
            &root,
            isolated_layer::CaptureGrid::ParentAligned,
        )
        .expect("bounded context");
        context.set_backing_extent(Size::new(128, 64));

        let region = primitive_region(
            &context,
            Rectangle::new(Point::new(10.0, 25.0), Size::new(50.0, 30.0)),
            Rectangle::from(Rectangle::with_size(context.backing_extent())),
        );

        assert_eq!(
            region,
            Some(primitive::RenderRegion::new(
                Rectangle::new(Point::new(10.0, 25.0), Size::new(50.0, 30.0)),
                Rectangle::new(Point::new(-20.0, -10.0), Size::new(100.0, 60.0)),
                Rectangle {
                    x: 0,
                    y: 0,
                    width: 80,
                    height: 50,
                },
            ))
        );
    }

    #[test]
    fn primitive_region_rejects_fully_clipped_bounds() {
        let context = root_context(Size::new(200, 100), 2.0);

        assert_eq!(
            primitive_region(
                &context,
                Rectangle::new(Point::new(150.0, 100.0), Size::new(10.0, 10.0)),
                Rectangle::from(Rectangle::with_size(context.physical_viewport())),
            ),
            None
        );
    }
}

fn clear_view(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("iced_wgpu isolated-layer clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn packed_color(color: Color) -> wgpu::Color {
    let [r, g, b, a] = graphics::color::pack(color).components();

    wgpu::Color {
        r: f64::from(r * a),
        g: f64::from(g * a),
        b: f64::from(b * a),
        a: f64::from(a),
    }
}

fn bounded_context(
    bounds: Rectangle,
    parent: &isolated_layer::Context,
    grid: isolated_layer::CaptureGrid,
    maximum_texture_dimension: u32,
) -> Option<isolated_layer::Context> {
    let mut context = isolated_layer::Context::bounded_with_grid(bounds, parent, grid)?;
    let backing_extent = isolated_layer::Pool::backing_extent(
        context.physical_viewport(),
        maximum_texture_dimension,
    )?;
    context.set_backing_extent(backing_extent);
    Some(context)
}

fn capture_grid(
    cache_requested: bool,
    positioning: core::isolated_layer::CompositePositioning,
) -> isolated_layer::CaptureGrid {
    if cache_requested || positioning == core::isolated_layer::CompositePositioning::Subpixel {
        isolated_layer::CaptureGrid::LayerAligned
    } else {
        isolated_layer::CaptureGrid::ParentAligned
    }
}

impl core::Renderer for Renderer {
    fn start_isolated_layer(&mut self, layer: core::isolated_layer::Layer) {
        self.recorder.start(&mut self.layers, layer);
    }

    fn end_isolated_layer(&mut self) {
        self.recorder.end(&mut self.layers);
    }

    fn mark_cache_alive(&self, keep_alive: core::isolated_layer::CacheKeepAlive) {
        self.isolated_layers.mark_cache_alive(keep_alive);
    }

    fn start_layer(&mut self, bounds: Rectangle) {
        self.layers.push_clip(bounds);
    }

    fn end_layer(&mut self) {
        self.layers.pop_clip();
    }

    fn start_transformation(&mut self, transformation: Transformation) {
        self.layers.push_transformation(transformation);
    }

    fn end_transformation(&mut self) {
        self.layers.pop_transformation();
    }

    fn fill_quad(&mut self, quad: core::renderer::Quad, background: impl Into<Background>) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_quad(quad, background.into(), transformation);
    }

    fn allocate_image(
        &self,
        _handle: &core::image::Handle,
        _callback: impl FnOnce(Result<core::image::Allocation, core::image::Error>) + Send + 'static,
    ) {
        #[cfg(feature = "image")]
        self.image_cache
            .borrow_mut()
            .allocate_image(_handle, _callback);
    }

    fn hint(&mut self, scale_factor: renderer::Scale) {
        self.scale = Some(scale_factor);
    }

    fn scale(&self) -> Option<renderer::Scale> {
        let scale_factor = self.scale?;

        Some(renderer::Scale {
            application: scale_factor.application * self.layers.transformation().scale_factor(),
            ..scale_factor
        })
    }

    fn tick(&mut self) {
        #[cfg(feature = "image")]
        self.image_cache.get_mut().receive();
    }

    fn reset(&mut self, new_bounds: Rectangle) {
        self.isolated_layers.snapshot_pending_keep_alives();
        self.recorder.reset();
        self.layers.reset(new_bounds);
    }

    fn settings(&self) -> renderer::Settings {
        self.settings
    }
}

impl core::text::Renderer for Renderer {
    type Font = Font;
    type Paragraph = Paragraph;
    type Editor = Editor;

    const ICON_FONT: Font = Font::new("Iced-Icons");
    const CHECKMARK_ICON: char = '\u{f00c}';
    const ARROW_DOWN_ICON: char = '\u{e800}';
    const ICED_LOGO: char = '\u{e801}';
    const SCROLL_UP_ICON: char = '\u{e802}';
    const SCROLL_DOWN_ICON: char = '\u{e803}';
    const SCROLL_LEFT_ICON: char = '\u{e804}';
    const SCROLL_RIGHT_ICON: char = '\u{e805}';

    fn default_font(&self) -> Self::Font {
        self.settings.default_font
    }

    fn default_size(&self) -> Pixels {
        self.settings.default_text_size
    }

    fn fill_paragraph(
        &mut self,
        text: &Self::Paragraph,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let (layer, transformation) = self.layers.current_mut();

        layer.draw_paragraph(text, position, color, clip_bounds, transformation);
    }

    fn fill_editor(
        &mut self,
        editor: &Self::Editor,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_editor(editor, position, color, clip_bounds, transformation);
    }

    fn fill_text(
        &mut self,
        text: core::Text,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_text(text, position, color, clip_bounds, transformation);
    }
}

impl graphics::text::Renderer for Renderer {
    fn fill_raw(&mut self, raw: graphics::text::Raw) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_text_raw(raw, transformation);
    }
}

#[cfg(feature = "image")]
impl core::image::Renderer for Renderer {
    type Handle = core::image::Handle;

    fn load_image(
        &self,
        handle: &Self::Handle,
    ) -> Result<core::image::Allocation, core::image::Error> {
        self.image_cache
            .borrow_mut()
            .load_image(&self.engine.device, &self.engine.queue, handle)
    }

    fn measure_image(&self, handle: &Self::Handle) -> Option<core::Size<u32>> {
        self.image_cache.borrow_mut().measure_image(handle)
    }

    fn draw_image(&mut self, image: core::Image, bounds: Rectangle, clip_bounds: Rectangle) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_raster(image, bounds, clip_bounds, transformation);
    }
}

#[cfg(feature = "svg")]
impl core::svg::Renderer for Renderer {
    fn measure_svg(&self, handle: &core::svg::Handle) -> core::Size<u32> {
        self.image_cache.borrow_mut().measure_svg(handle)
    }

    fn draw_svg(&mut self, svg: core::Svg, bounds: Rectangle, clip_bounds: Rectangle) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_svg(svg, bounds, clip_bounds, transformation);
    }
}

impl graphics::mesh::Renderer for Renderer {
    fn draw_mesh(&mut self, mesh: graphics::Mesh) {
        debug_assert!(
            !mesh.indices().is_empty(),
            "Mesh must not have empty indices"
        );

        debug_assert!(
            mesh.indices().len().is_multiple_of(3),
            "Mesh indices length must be a multiple of 3"
        );

        let (layer, transformation) = self.layers.current_mut();
        layer.draw_mesh(mesh, transformation);
    }

    fn draw_mesh_cache(&mut self, cache: mesh::Cache) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_mesh_cache(cache, transformation);
    }
}

#[cfg(feature = "geometry")]
impl graphics::geometry::Renderer for Renderer {
    type Geometry = Geometry;
    type Frame = geometry::Frame;

    fn new_frame(&self, bounds: Rectangle) -> Self::Frame {
        geometry::Frame::new(bounds)
    }

    fn draw_geometry(&mut self, geometry: Self::Geometry) {
        let (layer, transformation) = self.layers.current_mut();

        match geometry {
            Geometry::Live {
                meshes,
                images,
                text,
            } => {
                layer.draw_mesh_group(meshes, transformation);

                for image in images {
                    layer.draw_image(image, transformation);
                }

                layer.draw_text_group(text, transformation);
            }
            Geometry::Cached(cache) => {
                if let Some(meshes) = cache.meshes {
                    layer.draw_mesh_cache(meshes, transformation);
                }

                if let Some(images) = cache.images {
                    for image in images.iter().cloned() {
                        layer.draw_image(image, transformation);
                    }
                }

                if let Some(text) = cache.text {
                    layer.draw_text_cache(text, transformation);
                }
            }
        }
    }
}

impl primitive::Renderer for Renderer {
    fn draw_primitive(&mut self, bounds: Rectangle, primitive: impl Primitive) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_primitive(bounds, primitive, transformation);
    }
}

impl isolated_layer::Renderer for Renderer {
    fn start_isolated_layer_effects(
        &mut self,
        layer: isolated_layer::Layer,
        effects: isolated_layer::EffectStack,
    ) {
        self.recorder
            .start_effects(&mut self.layers, layer, effects);
    }

    fn end_isolated_layer_effects(&mut self) {
        self.recorder.end(&mut self.layers);
    }
}

impl graphics::compositor::Default for crate::Renderer {
    type Compositor = window::Compositor;
}

impl renderer::Headless for Renderer {
    async fn new(settings: renderer::Settings, backend: Option<&str>) -> Option<Self> {
        if backend.is_some_and(|backend| backend != "wgpu") {
            return None;
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("iced_wgpu [headless]"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_bind_groups: 2,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .ok()?;

        let engine = Engine::new(
            &adapter,
            device,
            queue,
            if graphics::color::GAMMA_CORRECTION {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            Some(graphics::Antialiasing::MSAAx4),
            Shell::headless(),
        );

        Some(Self::new(engine, settings))
    }

    fn name(&self) -> String {
        "wgpu".to_owned()
    }

    fn screenshot(
        &mut self,
        size: Size<u32>,
        scale_factor: f32,
        background_color: Color,
    ) -> Vec<u8> {
        self.screenshot(
            &Viewport::with_physical_size(
                size,
                renderer::Scale {
                    window: 1.0,
                    application: scale_factor,
                },
            ),
            background_color,
        )
    }
}
