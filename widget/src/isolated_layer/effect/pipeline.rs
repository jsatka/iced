use crate::renderer::wgpu::isolated_layer::effect;
use crate::renderer::wgpu::shader::isolated_layer::effect as shader;
use crate::renderer::wgpu::wgpu;

use bytemuck::{Pod, Zeroable};
use std::any::Any;
use wgpu::util::DeviceExt;

// Match `LayerEffect`'s platform-specific `MaybeSend` and `MaybeSync` bounds.
#[cfg(not(target_arch = "wasm32"))]
pub(super) type PreparedPass = Box<dyn Any + Send + Sync>;

#[cfg(target_arch = "wasm32")]
pub(super) type PreparedPass = Box<dyn Any>;

pub(super) struct Prepared {
    bind_group: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
}

pub(super) struct BlurPipeline(pub(super) TexturePipeline);

pub(super) struct TexturePipeline {
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl effect::Pipeline for BlurPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self(TexturePipeline::new(
            device,
            format,
            "iced_widget.isolated_layer.blur",
            shader::BLUR,
            &[
                texture_entry(0),
                sampler_entry(1),
                uniform_entry::<BlurParams>(2),
            ],
        ))
    }
}

impl TexturePipeline {
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        family: &str,
        source: &'static str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{family}.sampler")),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{family}.bind_group_layout")),
            entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{family}.shader")),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{family}.pipeline_layout")),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{family}.pipeline")),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            sampler,
            layout,
            pipeline,
        }
    }

    pub(super) fn prepare<T: Pod>(
        &self,
        device: &wgpu::Device,
        family: &str,
        params: &T,
        textures: &[(u32, &wgpu::TextureView)],
        sampler_binding: u32,
        uniform_binding: u32,
    ) -> Prepared {
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{family}.uniform")),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries: Vec<_> = textures
            .iter()
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: sampler_binding,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: uniform_binding,
            resource: uniform.as_entire_binding(),
        });
        entries.sort_unstable_by_key(|entry| entry.binding);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{family}.bind_group")),
            layout: &self.layout,
            entries: &entries,
        });

        Prepared {
            bind_group,
            _uniform: uniform,
        }
    }

    pub(super) fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: crate::core::Size<u32>,
        family: &str,
        prepared: &Prepared,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!("{family}.pass")),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, size.width as f32, size.height as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, size.width, size.height);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &prepared.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct BlurParams {
    geometry: [f32; 4],
    parameters: [f32; 4],
}

impl BlurParams {
    pub(super) fn new(context: &effect::Context, pass: usize, sigma: f32) -> Self {
        Self::for_input(
            context,
            context.physical_size,
            pass,
            sigma * context.scale_factor,
        )
    }

    pub(super) fn for_input(
        context: &effect::Context,
        input_size: crate::core::Size<u32>,
        pass: usize,
        sigma: f32,
    ) -> Self {
        Self {
            geometry: geometry_for_size(context, input_size),
            parameters: [
                if pass == 0 { 1.0 } else { 0.0 },
                if pass == 0 { 0.0 } else { 1.0 },
                sigma,
                0.0,
            ],
        }
    }
}

pub(super) fn geometry(context: &effect::Context) -> [f32; 4] {
    geometry_for_size(context, context.physical_size)
}

pub(super) fn geometry_for_size(
    context: &effect::Context,
    size: crate::core::Size<u32>,
) -> [f32; 4] {
    [
        size.width as f32 / context.backing_extent.width as f32,
        size.height as f32 / context.backing_extent.height as f32,
        1.0 / context.backing_extent.width as f32,
        1.0 / context.backing_extent.height as f32,
    ]
}

pub(super) fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

pub(super) fn uniform_entry<T>(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<T>() as u64),
        },
        count: None,
    }
}
