use super::canonical;
use super::pipeline::{BlurAxis, BlurParams, BlurPipeline, Prepared};
use crate::core::Padding;
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Plan, Requirements, TextureViews,
};
use crate::renderer::wgpu::wgpu;

/// Gaussian blur settings and effect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianBlur {
    /// The standard deviation of the blur kernel in logical pixels.
    pub sigma: f32,
}

impl GaussianBlur {
    /// Creates Gaussian blur settings.
    pub fn new(sigma: f32) -> Self {
        Self {
            sigma: canonical(sigma, 0.0, 128.0),
        }
    }
}

impl Default for GaussianBlur {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl effect::LayerEffect for GaussianBlur {
    fn plan(&self, plan: &mut Plan<'_, Self>) {
        plan.push(BlurPass(BlurAxis::Horizontal));
        plan.push(BlurPass(BlurAxis::Vertical));
    }

    fn expansion(&self) -> Padding {
        Padding::new((self.sigma * 3.0).ceil())
    }

    fn is_translation_invariant(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlurPass(BlurAxis);

impl effect::Pass<GaussianBlur> for BlurPass {
    type Prepared = Prepared;

    fn requirements(&self, _effect: &GaussianBlur) -> Requirements {
        Requirements::new().writes_every_pixel()
    }

    fn prepare(
        &self,
        effect: &GaussianBlur,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Prepared {
        let pipeline = pipelines.get_or_init::<BlurPipeline>();
        pipeline.0.prepare(
            device,
            "iced_widget.isolated_layer.blur",
            &BlurParams::new(context, self.0, effect.sigma),
            &[(0, views.previous)],
            1,
            2,
        )
    }

    fn encode(
        &self,
        _effect: &GaussianBlur,
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
