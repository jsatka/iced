use super::gaussian_blur::{BlurAxis, BlurPipeline};
use super::pipeline::{Prepared, TexturePipeline, sampler_entry, texture_entry, uniform_entry};
use super::{Blur, dual_kawase_blur::Chain};
use crate::core::{Color, Padding, Vector};
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Plan, Requirements, TextureViews,
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
    /// Blur algorithm and settings. Gaussian is the default.
    pub blur: Blur,
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
    fn plan(&self, plan: &mut Plan<'_, Self>) {
        match self.blur {
            Blur::Gaussian(settings) => {
                if settings.normalized().radius > 0.0 {
                    plan.push(BlurPass(BlurAxis::Horizontal));
                    plan.push(BlurPass(BlurAxis::Vertical));
                }
                plan.push(ShadowPass);
            }
            Blur::DualKawase(_) => plan.push(KawaseShadowPass),
        }
    }

    fn expansion(&self) -> Padding {
        let blur = match self.blur {
            Blur::Gaussian(settings) => settings.padding(),
            Blur::DualKawase(settings) => settings.padding(),
        };
        Padding {
            top: blur + (-self.offset.y).max(0.0),
            right: blur + self.offset.x.max(0.0),
            bottom: blur + self.offset.y.max(0.0),
            left: blur + (-self.offset.x).max(0.0),
        }
    }

    fn record_inputs(&self, inputs: &mut effect::LayerInputRecords) {
        let mut settings = *self;
        if let Blur::Gaussian(blur) = settings.blur {
            settings.blur = Blur::Gaussian(blur.normalized());
        }
        inputs.record(&settings);
    }

    fn is_translation_invariant(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlurPass(BlurAxis);

impl effect::Pass<DropShadow> for BlurPass {
    type Prepared = Prepared;

    fn requirements(&self, _effect: &DropShadow) -> Requirements {
        Requirements::new().writes_every_pixel()
    }

    fn prepare(
        &self,
        effect: &DropShadow,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _scratch: &mut effect::ScratchAllocator<'_>,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Prepared {
        let Blur::Gaussian(settings) = effect.blur else {
            unreachable!("Gaussian pass belongs to a Gaussian shadow plan");
        };
        let pipeline = pipelines.get_or_init::<BlurPipeline>();
        pipeline.prepare(
            device,
            queue,
            "iced_widget.isolated_layer.blur",
            context,
            self.0,
            settings,
            views.previous,
        )
    }

    fn encode(
        &self,
        _effect: &DropShadow,
        pipelines: &PipelineRegistry<'_>,
        prepared: &Prepared,
        encoder: &mut wgpu::CommandEncoder,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) {
        let pipeline = pipelines.get::<BlurPipeline>().expect("blur pipeline");
        pipeline.0.render(
            encoder,
            views.output,
            context.physical_size,
            "iced_widget.isolated_layer.blur",
            prepared,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShadowPass;

impl effect::Pass<DropShadow> for ShadowPass {
    type Prepared = Prepared;

    fn requirements(&self, _effect: &DropShadow) -> Requirements {
        Requirements::new().writes_every_pixel()
    }

    fn prepare(
        &self,
        effect: &DropShadow,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _scratch: &mut effect::ScratchAllocator<'_>,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Prepared {
        let pipeline = pipelines.get_or_init::<ShadowPipeline>();
        pipeline.0.prepare(
            device,
            "iced_widget.isolated_layer.shadow",
            &ShadowParams::new(context, *effect),
            &[(0, views.stage_input), (1, views.previous)],
            2,
            3,
        )
    }

    fn encode(
        &self,
        _effect: &DropShadow,
        pipelines: &PipelineRegistry<'_>,
        prepared: &Prepared,
        encoder: &mut wgpu::CommandEncoder,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) {
        let pipeline = pipelines.get::<ShadowPipeline>().expect("shadow pipeline");
        pipeline.0.render(
            encoder,
            views.output,
            context.physical_size,
            "iced_widget.isolated_layer.shadow",
            prepared,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KawaseShadowPass;

impl effect::Pass<DropShadow> for KawaseShadowPass {
    type Prepared = Chain;

    fn requirements(&self, _effect: &DropShadow) -> Requirements {
        Requirements::new().writes_every_pixel()
    }

    fn prepare(
        &self,
        effect: &DropShadow,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        scratch: &mut effect::ScratchAllocator<'_>,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Chain {
        let Blur::DualKawase(settings) = effect.blur else {
            unreachable!("Kawase pass belongs to a Kawase shadow plan");
        };
        Chain::prepare(
            settings,
            Some((effect.color, effect.offset)),
            pipelines,
            device,
            scratch,
            context,
            views,
        )
    }

    fn encode(
        &self,
        _effect: &DropShadow,
        pipelines: &PipelineRegistry<'_>,
        prepared: &Chain,
        encoder: &mut wgpu::CommandEncoder,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) {
        prepared.encode(pipelines, encoder, context, views.output);
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
