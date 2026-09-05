use super::{Context, Placement, Target};
use crate::core::isolated_layer::{BlendMode, Composite, CompositePositioning};
use crate::core::{Point, Rectangle, Size};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Prepared renderer-owned final composition resources.
pub(crate) struct Prepared {
    positioning: CompositePositioning,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
}

/// Empty-at-start storage for lazily-created composition variants.
#[derive(Default)]
pub(crate) struct Storage {
    snapped_common: Option<Common>,
    snapped_source_over: Option<wgpu::RenderPipeline>,
    snapped_add: Option<wgpu::RenderPipeline>,
    subpixel_common: Option<SubpixelCommon>,
    subpixel_source_over: Option<wgpu::RenderPipeline>,
    subpixel_add: Option<wgpu::RenderPipeline>,
    subpixel_replace: Option<wgpu::RenderPipeline>,
}

struct Common {
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
}

struct SubpixelCommon {
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
}

#[derive(Debug, Clone, Copy)]
enum SubpixelBlend {
    SourceOver,
    Add,
    Replace,
}

/// Complete source-to-destination mapping for one subpixel blit.
///
/// Keeping these values together makes the coordinate-space contract explicit at both call sites:
/// final composition and backdrop acquisition.
#[derive(Debug, Clone, Copy)]
struct SubpixelBlit {
    valid_extent: Size<u32>,
    backing_extent: Size<u32>,
    source_origin: Point,
    destination_origin: Point,
    opacity: f32,
    blend: SubpixelBlend,
}

impl SubpixelBlit {
    fn composite(context: &Context, composite: Composite) -> Self {
        Self {
            valid_extent: context.physical_viewport(),
            backing_extent: context.backing_extent(),
            source_origin: Point::ORIGIN,
            destination_origin: context.placement.exact_origin(),
            opacity: composite.opacity(),
            blend: match composite.blend_mode() {
                BlendMode::SourceOver => SubpixelBlend::SourceOver,
                BlendMode::Add => SubpixelBlend::Add,
            },
        }
    }

    fn backdrop(parent: &Context, child: &Context) -> Self {
        Self {
            valid_extent: parent.physical_viewport(),
            backing_extent: parent.backing_extent(),
            source_origin: child.placement.exact_origin(),
            destination_origin: Point::ORIGIN,
            opacity: 1.0,
            blend: SubpixelBlend::Replace,
        }
    }

    fn debug_assert_valid(self) {
        debug_assert!(self.valid_extent.width <= self.backing_extent.width);
        debug_assert!(self.valid_extent.height <= self.backing_extent.height);
        debug_assert!(self.source_origin.x.is_finite() && self.source_origin.y.is_finite());
        debug_assert!(
            self.destination_origin.x.is_finite() && self.destination_origin.y.is_finite()
        );
        debug_assert!(self.opacity.is_finite() && (0.0..=1.0).contains(&self.opacity));
    }
}

impl Storage {
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        source: &Target,
        context: &Context,
        composite: Composite,
    ) -> Prepared {
        match composite.positioning() {
            CompositePositioning::Snapped => {
                self.prepare_snapped(device, format, source, context, composite)
            }
            CompositePositioning::Subpixel => self.prepare_subpixel(
                device,
                format,
                &source.view,
                SubpixelBlit::composite(context, composite),
            ),
        }
    }

    fn prepare_snapped(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        source: &Target,
        context: &Context,
        composite: Composite,
    ) -> Prepared {
        let common = self
            .snapped_common
            .get_or_insert_with(|| Common::new(device));
        let slot = match composite.blend_mode() {
            BlendMode::SourceOver => &mut self.snapped_source_over,
            BlendMode::Add => &mut self.snapped_add,
        };
        let pipeline = slot
            .get_or_insert_with(|| render_pipeline(device, common, format, composite.blend_mode()))
            .clone();

        let params = Params {
            geometry: [
                context.valid_uv()[0],
                context.valid_uv()[1],
                1.0 / context.backing_extent().width as f32,
                1.0 / context.backing_extent().height as f32,
            ],
            opacity: [composite.opacity(), 0.0, 0.0, 0.0],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iced_wgpu.isolated_layer.composite.uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iced_wgpu.isolated_layer.composite.bind_group"),
            layout: &common.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&common.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        Prepared {
            positioning: CompositePositioning::Snapped,
            pipeline,
            bind_group,
            _uniform: uniform,
        }
    }

    pub fn prepare_backdrop(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        source: &wgpu::TextureView,
        parent: &Context,
        child: &Context,
    ) -> Prepared {
        self.prepare_subpixel(
            device,
            format,
            source,
            SubpixelBlit::backdrop(parent, child),
        )
    }

    fn prepare_subpixel(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        source: &wgpu::TextureView,
        blit: SubpixelBlit,
    ) -> Prepared {
        blit.debug_assert_valid();

        let common = self
            .subpixel_common
            .get_or_insert_with(|| SubpixelCommon::new(device));
        let slot = match blit.blend {
            SubpixelBlend::SourceOver => &mut self.subpixel_source_over,
            SubpixelBlend::Add => &mut self.subpixel_add,
            SubpixelBlend::Replace => &mut self.subpixel_replace,
        };
        let pipeline = slot
            .get_or_insert_with(|| subpixel_pipeline(device, common, format, blit.blend))
            .clone();
        let params = SubpixelParams {
            origins: [
                blit.source_origin.x,
                blit.source_origin.y,
                blit.destination_origin.x,
                blit.destination_origin.y,
            ],
            source_geometry: [
                blit.valid_extent.width as f32,
                blit.valid_extent.height as f32,
                blit.backing_extent.width as f32,
                blit.backing_extent.height as f32,
            ],
            opacity: [blit.opacity, 0.0, 0.0, 0.0],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iced_wgpu.isolated_layer.subpixel_blit.uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iced_wgpu.isolated_layer.subpixel_blit.bind_group"),
            layout: &common.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&common.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        Prepared {
            positioning: CompositePositioning::Subpixel,
            pipeline,
            bind_group,
            _uniform: uniform,
        }
    }
}

impl Common {
    fn new(device: &wgpu::Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("iced_wgpu.isolated_layer.composite.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iced_wgpu.isolated_layer.composite.bind_group_layout"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Params>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iced_wgpu.isolated_layer.composite.shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shader/isolated_layer/composite.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iced_wgpu.isolated_layer.composite.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        Self {
            sampler,
            layout,
            shader,
            pipeline_layout,
        }
    }
}

impl SubpixelCommon {
    fn new(device: &wgpu::Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("iced_wgpu.isolated_layer.subpixel_blit.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iced_wgpu.isolated_layer.subpixel_blit.bind_group_layout"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SubpixelParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iced_wgpu.isolated_layer.subpixel_blit.shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shader/isolated_layer/subpixel_blit.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iced_wgpu.isolated_layer.subpixel_blit.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        Self {
            sampler,
            layout,
            shader,
            pipeline_layout,
        }
    }
}

pub(crate) fn render(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    parent_size: Size<u32>,
    placement: Placement,
    clip: Rectangle<u32>,
    load: wgpu::LoadOp<wgpu::Color>,
    prepared: &Prepared,
) {
    let destination = placement.snapped();
    let scissor = match prepared.positioning {
        CompositePositioning::Snapped => intersect(destination, clip),
        CompositePositioning::Subpixel => placement
            .conservative_coverage(parent_size)
            .and_then(|coverage| intersect(coverage, clip)),
    };
    let Some(scissor) = scissor else {
        return;
    };

    render_prepared(
        encoder,
        target,
        parent_size,
        destination,
        scissor,
        load,
        prepared,
    );
}

pub(crate) fn render_backdrop(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    size: Size<u32>,
    prepared: &Prepared,
) {
    debug_assert_eq!(prepared.positioning, CompositePositioning::Subpixel);

    let bounds = Rectangle::with_size(size);
    render_prepared(
        encoder,
        target,
        size,
        bounds,
        bounds,
        wgpu::LoadOp::Load,
        prepared,
    );
}

fn render_prepared(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    viewport: Size<u32>,
    destination: Rectangle<u32>,
    scissor: Rectangle<u32>,
    load: wgpu::LoadOp<wgpu::Color>,
    prepared: &Prepared,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("iced_wgpu.isolated_layer.composite.pass"),
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
    match prepared.positioning {
        CompositePositioning::Snapped => pass.set_viewport(
            destination.x as f32,
            destination.y as f32,
            destination.width as f32,
            destination.height as f32,
            0.0,
            1.0,
        ),
        CompositePositioning::Subpixel => pass.set_viewport(
            0.0,
            0.0,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        ),
    }
    pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
    pass.set_pipeline(&prepared.pipeline);
    pass.set_bind_group(0, &prepared.bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn intersect(a: Rectangle<u32>, b: Rectangle<u32>) -> Option<Rectangle<u32>> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = a.x.saturating_add(a.width).min(b.x.saturating_add(b.width));
    let bottom =
        a.y.saturating_add(a.height)
            .min(b.y.saturating_add(b.height));

    (right > x && bottom > y).then_some(Rectangle {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

fn render_pipeline(
    device: &wgpu::Device,
    common: &Common,
    format: wgpu::TextureFormat,
    blend_mode: BlendMode,
) -> wgpu::RenderPipeline {
    let (label, blend) = match blend_mode {
        BlendMode::SourceOver => (
            "iced_wgpu.isolated_layer.composite.source_over.pipeline",
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        ),
        BlendMode::Add => (
            "iced_wgpu.isolated_layer.composite.add.pipeline",
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
        ),
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&common.pipeline_layout),
        vertex: wgpu::VertexState {
            module: &common.shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &common.shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn subpixel_pipeline(
    device: &wgpu::Device,
    common: &SubpixelCommon,
    format: wgpu::TextureFormat,
    blend: SubpixelBlend,
) -> wgpu::RenderPipeline {
    let (label, blend) = match blend {
        SubpixelBlend::SourceOver => (
            "iced_wgpu.isolated_layer.subpixel_blit.source_over.pipeline",
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        ),
        SubpixelBlend::Add => (
            "iced_wgpu.isolated_layer.subpixel_blit.add.pipeline",
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
        ),
        SubpixelBlend::Replace => (
            "iced_wgpu.isolated_layer.subpixel_blit.replace.pipeline",
            wgpu::BlendState::REPLACE,
        ),
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&common.pipeline_layout),
        vertex: wgpu::VertexState {
            module: &common.shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &common.shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    geometry: [f32; 4],
    opacity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SubpixelParams {
    origins: [f32; 4],
    source_geometry: [f32; 4],
    opacity: [f32; 4],
}

#[cfg(test)]
mod tests {
    fn edge_weight(position: f32, extent: f32) -> f32 {
        (position.min(extent - position) + 0.5).clamp(0.0, 1.0)
    }

    #[test]
    fn transparent_border_weights_are_exact_at_edges_and_centers() {
        let extent = 8.0;

        assert_eq!(edge_weight(-0.5, extent), 0.0);
        assert_eq!(edge_weight(-0.25, extent), 0.25);
        assert_eq!(edge_weight(0.0, extent), 0.5);
        assert_eq!(edge_weight(0.25, extent), 0.75);
        assert_eq!(edge_weight(0.5, extent), 1.0);
        assert_eq!(edge_weight(4.5, extent), 1.0);
        assert_eq!(edge_weight(7.5, extent), 1.0);
        assert_eq!(edge_weight(8.0, extent), 0.5);
        assert_eq!(edge_weight(8.25, extent), 0.25);
        assert_eq!(edge_weight(8.5, extent), 0.0);
    }

    #[test]
    fn corner_coverage_factors_into_axis_weights() {
        assert_eq!(edge_weight(-0.25, 8.0) * edge_weight(0.25, 8.0), 0.1875);
        assert_eq!(edge_weight(0.5, 8.0) * edge_weight(7.5, 8.0), 1.0);
    }
}
