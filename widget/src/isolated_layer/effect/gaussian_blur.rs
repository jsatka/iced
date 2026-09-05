use super::canonical;
use super::pipeline::{BlurParams, BlurPipeline, Prepared, PreparedPass};
use crate::core::Padding;
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Requirements, TextureViews,
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
    type PreparedPass = PreparedPass;

    fn requirements(&self) -> Requirements {
        Requirements::passes(2).writes_every_pixel()
    }

    fn expansion(&self) -> Padding {
        Padding::new((self.sigma * 3.0).ceil())
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
        let pipeline = pipelines.get_or_init::<BlurPipeline>();
        let input = if pass == 0 {
            views.stage_input
        } else {
            views.previous
        };
        Box::new(pipeline.0.prepare(
            device,
            "iced_widget.isolated_layer.blur",
            &BlurParams::new(context, pass, self.sigma),
            &[(0, input)],
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
        let pipeline = pipelines.get::<BlurPipeline>().expect("blur pipeline");
        let prepared = prepared
            .as_ref()
            .downcast_ref::<Prepared>()
            .expect("blur pass");
        pipeline.0.render(
            encoder,
            views.output,
            context.physical_size,
            "iced_widget.isolated_layer.blur",
            prepared,
        );
    }
}
