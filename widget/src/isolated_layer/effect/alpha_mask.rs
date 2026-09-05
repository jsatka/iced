use super::canonical;
use super::pipeline::{
    Prepared, PreparedPass, TexturePipeline, sampler_entry, texture_entry, uniform_entry,
};
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Requirements, TextureViews,
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
    type PreparedPass = PreparedPass;

    fn requirements(&self) -> Requirements {
        Requirements::passes(1).writes_every_pixel()
    }

    fn is_translation_invariant(&self) -> bool {
        true
    }

    fn prepare_pass(
        &self,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _pass: usize,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> PreparedPass {
        let pipeline = pipelines.get_or_init::<MaskPipeline>();
        Box::new(pipeline.0.prepare(
            device,
            "iced_widget.isolated_layer.mask",
            &MaskParams::new(context, *self),
            &[(0, views.stage_input)],
            1,
            2,
        ))
    }

    fn encode_pass(
        &self,
        pipelines: &PipelineRegistry<'_>,
        prepared: &PreparedPass,
        encoder: &mut wgpu::CommandEncoder,
        _pass: usize,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) {
        let pipeline = pipelines.get::<MaskPipeline>().expect("mask pipeline");
        let prepared = prepared
            .as_ref()
            .downcast_ref::<Prepared>()
            .expect("mask pass");
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
