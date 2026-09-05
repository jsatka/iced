use super::pipeline::{
    BlurParams, BlurPipeline, Prepared, PreparedPass, TexturePipeline, sampler_entry,
    texture_entry, uniform_entry,
};
use crate::core::{Color, Padding, Vector};
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Requirements, TextureViews,
};
use crate::renderer::wgpu::shader::isolated_layer::effect as shader;
use crate::renderer::wgpu::wgpu;

use bytemuck::{Pod, Zeroable};

/// Drop shadow settings and effect.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DropShadow {
    /// The color of the shadow.
    pub color: Color,
    /// The offset of the shadow.
    pub offset: Vector,
    /// The blur radius of the shadow.
    pub blur_radius: f32,
}

enum Pass {
    Blur(Prepared),
    Shadow(Prepared),
}

struct ShadowPipeline(TexturePipeline);

impl effect::Pipeline for ShadowPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self(TexturePipeline::new(
            device,
            format,
            "iced_widget.isolated_layer.shadow",
            shader::SHADOW,
            &[
                texture_entry(0),
                texture_entry(1),
                sampler_entry(2),
                uniform_entry::<ShadowParams>(3),
            ],
        ))
    }
}

impl effect::LayerEffect for DropShadow {
    type PreparedPass = PreparedPass;

    fn requirements(&self) -> Requirements {
        Requirements::passes(3).writes_every_pixel()
    }

    fn expansion(&self) -> Padding {
        let blur = (self.blur_radius * 3.0).ceil();
        Padding {
            top: blur + (-self.offset.y).max(0.0),
            right: blur + self.offset.x.max(0.0),
            bottom: blur + self.offset.y.max(0.0),
            left: blur + (-self.offset.x).max(0.0),
        }
    }

    fn is_translation_invariant(&self) -> bool {
        true
    }

    fn prepare_pass(
        &self,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        pass: usize,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> PreparedPass {
        if pass < 2 {
            let pipeline = pipelines.get_or_init::<BlurPipeline>();
            let input = if pass == 0 {
                views.stage_input
            } else {
                views.previous
            };
            Box::new(Pass::Blur(pipeline.0.prepare(
                device,
                "iced_widget.isolated_layer.blur",
                &BlurParams::new(context, pass, self.blur_radius),
                &[(0, input)],
                1,
                2,
            )))
        } else {
            let pipeline = pipelines.get_or_init::<ShadowPipeline>();
            Box::new(Pass::Shadow(pipeline.0.prepare(
                device,
                "iced_widget.isolated_layer.shadow",
                &ShadowParams::new(context, *self),
                &[(0, views.stage_input), (1, views.previous)],
                2,
                3,
            )))
        }
    }

    fn encode_pass(
        &self,
        pipelines: &PipelineRegistry<'_>,
        prepared: &PreparedPass,
        encoder: &mut wgpu::CommandEncoder,
        pass: usize,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) {
        if pass < 2 {
            let pipeline = pipelines.get::<BlurPipeline>().expect("blur pipeline");
            let Pass::Blur(prepared) = prepared.as_ref().downcast_ref::<Pass>().expect("blur pass")
            else {
                unreachable!()
            };
            pipeline.0.render(
                encoder,
                views.output,
                context.physical_size,
                "iced_widget.isolated_layer.blur",
                prepared,
            );
        } else {
            let pipeline = pipelines.get::<ShadowPipeline>().expect("shadow pipeline");
            let Pass::Shadow(prepared) = prepared
                .as_ref()
                .downcast_ref::<Pass>()
                .expect("shadow pass")
            else {
                unreachable!()
            };
            pipeline.0.render(
                encoder,
                views.output,
                context.physical_size,
                "iced_widget.isolated_layer.shadow",
                prepared,
            );
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowParams {
    geometry: [f32; 4],
    parameters: [f32; 4],
    color: [f32; 4],
}

impl ShadowParams {
    fn new(context: &effect::Context, shadow: DropShadow) -> Self {
        Self {
            geometry: super::pipeline::geometry(context),
            parameters: [
                shadow.offset.x * context.scale_factor,
                shadow.offset.y * context.scale_factor,
                0.0,
                0.0,
            ],
            color: crate::graphics::color::pack(shadow.color).components(),
        }
    }
}
