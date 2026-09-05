use super::canonical;
use super::pipeline::{Prepared, TexturePipeline, sampler_entry, texture_entry, uniform_entry};
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Plan, Requirements, TextureViews,
};
use crate::renderer::wgpu::shader::isolated_layer::effect as shader;
use crate::renderer::wgpu::wgpu;

use bytemuck::{Pod, Zeroable};

/// A generated alpha mask for soft container edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaMask {
    /// Top fade distance in logical pixels.
    pub top: f32,
    /// Right fade distance in logical pixels.
    pub right: f32,
    /// Bottom fade distance in logical pixels.
    pub bottom: f32,
    /// Left fade distance in logical pixels.
    pub left: f32,
}

impl AlphaMask {
    /// Creates an edge-fade alpha mask.
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self {
            top: canonical(top, 0.0, 4_096.0),
            right: canonical(right, 0.0, 4_096.0),
            bottom: canonical(bottom, 0.0, 4_096.0),
            left: canonical(left, 0.0, 4_096.0),
        }
    }

    /// Creates a mask that fades only the top and bottom edges.
    pub fn vertical(top: f32, bottom: f32) -> Self {
        Self::new(top, 0.0, bottom, 0.0)
    }
}

impl Default for AlphaMask {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

struct MaskPipeline(TexturePipeline);

impl effect::Pipeline for MaskPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self(TexturePipeline::new(
            device,
            format,
            "iced_widget.isolated_layer.mask",
            shader::MASK,
            &[
                texture_entry(0),
                sampler_entry(1),
                uniform_entry::<MaskParams>(2),
            ],
        ))
    }
}

impl effect::LayerEffect for AlphaMask {
    fn plan(&self, plan: &mut Plan<'_, Self>) {
        plan.push(MaskPass);
    }

    fn is_translation_invariant(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskPass;

impl effect::Pass<AlphaMask> for MaskPass {
    type Prepared = Prepared;

    fn requirements(&self, _effect: &AlphaMask) -> Requirements {
        Requirements::new().writes_every_pixel()
    }

    fn prepare(
        &self,
        effect: &AlphaMask,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Prepared {
        let pipeline = pipelines.get_or_init::<MaskPipeline>();
        pipeline.0.prepare(
            device,
            "iced_widget.isolated_layer.mask",
            &MaskParams::new(context, *effect),
            &[(0, views.stage_input)],
            1,
            2,
        )
    }

    fn encode(
        &self,
        _effect: &AlphaMask,
        pipelines: &PipelineRegistry<'_>,
        prepared: &Prepared,
        encoder: &mut wgpu::CommandEncoder,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) {
        let pipeline = pipelines.get::<MaskPipeline>().expect("mask pipeline");
        pipeline.0.render(
            encoder,
            views.output,
            context.physical_size,
            "iced_widget.isolated_layer.mask",
            prepared,
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskParams {
    geometry: [f32; 4],
    viewport: [f32; 4],
    edges: [f32; 4],
}

impl MaskParams {
    fn new(context: &effect::Context, mask: AlphaMask) -> Self {
        Self {
            geometry: super::pipeline::geometry(context),
            viewport: [
                context.physical_size.width as f32,
                context.physical_size.height as f32,
                0.0,
                0.0,
            ],
            edges: [mask.top, mask.right, mask.bottom, mask.left]
                .map(|value| value * context.scale_factor),
        }
    }
}
