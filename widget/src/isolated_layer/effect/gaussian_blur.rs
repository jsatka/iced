use super::canonical;
use super::pipeline::{
    Prepared, TexturePipeline, geometry, sampler_entry, texture_entry, uniform_entry,
};
use crate::core::Padding;
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Plan, Requirements, TextureViews,
};
use crate::renderer::wgpu::shader::isolated_layer::effect as shader;
use crate::renderer::wgpu::wgpu;

use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;
use std::sync::Arc;

/// Gaussian blur settings and effect.
///
/// Uses two full-resolution passes over consecutive physical pixels, with a
/// standard deviation of `radius / 3`. The kernel grows with radius and display
/// scale. Zero radius forwards the input without any blur passes.
///
/// Large radii and high display scales are expensive; use [`super::DualKawaseBlur`]
/// when a cheaper approximation is preferred.
/// To migrate an old sigma setting, use `radius = (3.0 * sigma).min(128.0)`;
/// the dense kernel may change the appearance of the former sparse blur.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianBlur {
    /// The approximate outer radius in logical pixels, clamped to 0–128.
    ///
    /// Negative values and NaN become zero; positive infinity becomes 128.
    /// This also applies when settings are constructed using public fields.
    pub radius: f32,
}

impl GaussianBlur {
    /// Creates Gaussian blur settings.
    pub fn new(radius: f32) -> Self {
        Self {
            radius: canonical(radius, 0.0, 128.0),
        }
    }

    pub(super) fn normalized(self) -> Self {
        Self::new(self.radius)
    }

    pub(super) fn padding(self) -> f32 {
        self.normalized().radius.ceil()
    }
}

impl Default for GaussianBlur {
    fn default() -> Self {
        Self::new(0.0)
    }
}

pub(super) struct BlurPipeline(pub(super) TexturePipeline, VecDeque<Arc<CachedKernel>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlurAxis {
    Horizontal,
    Vertical,
}

impl effect::Pipeline for BlurPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self(
            TexturePipeline::new(
                device,
                format,
                "iced_widget.isolated_layer.blur",
                shader::GAUSSIAN,
                &[
                    texture_entry(0),
                    sampler_entry(1),
                    uniform_entry::<BlurParams>(2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            ),
            VecDeque::new(),
        )
    }
}

impl BlurPipeline {
    // Bound animated-radius cache growth. Entries are immutable: evicting a
    // kernel never overwrites coefficients used by already prepared passes.
    const CACHED_KERNELS: usize = 8;

    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        family: &str,
        context: &effect::Context,
        axis: BlurAxis,
        settings: GaussianBlur,
        source: &wgpu::TextureView,
    ) -> Prepared {
        let radius = settings.normalized().radius * context.scale_factor;
        let kernel = self.kernel(device, queue, radius);
        let params = BlurParams::new(context, axis, kernel.center, kernel.pair_count);
        self.0.prepare(
            device,
            family,
            &params,
            &[(0, source), (3, &kernel.view)],
            1,
            2,
        )
    }

    fn kernel(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        radius: f32,
    ) -> Arc<CachedKernel> {
        if let Some(index) = self.1.iter().position(|kernel| kernel.radius == radius) {
            let kernel = self.1.remove(index).expect("cached Gaussian kernel");
            self.1.push_back(Arc::clone(&kernel));
            return kernel;
        }

        // A valid expanded layer must fit its complete diameter in a target.
        assert!(
            radius.is_finite()
                && radius >= 0.0
                && radius <= device.limits().max_texture_dimension_2d as f32,
            "Gaussian radius must fit the layer target"
        );

        let mut coefficients = GaussianKernel::new(radius);
        let pair_count = coefficients.pairs.len() as u32;
        let width = pair_count.max(1);
        coefficients.pairs.resize(width as usize, [0.0; 2]);

        // A coefficient texture supports variable kernels on WebGL as well as
        // native backends, without fixed uniform-array limits or storage buffers.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_widget.isolated_layer.blur.kernel"),
            size: wgpu::Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&coefficients.pairs),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: None,
            },
            texture.size(),
        );

        let kernel = Arc::new(CachedKernel {
            radius,
            center: coefficients.center,
            pair_count,
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        });
        if self.1.len() == Self::CACHED_KERNELS {
            let _ = self.1.pop_front();
        }
        self.1.push_back(Arc::clone(&kernel));
        kernel
    }
}

struct CachedKernel {
    radius: f32,
    center: f32,
    pair_count: u32,
    view: wgpu::TextureView,
}

struct GaussianKernel {
    center: f32,
    // Positive-side adjacent taps combined into [physical offset, total weight].
    pairs: Vec<[f32; 2]>,
}

impl GaussianKernel {
    fn new(physical_radius: f32) -> Self {
        if physical_radius == 0.0 {
            return Self {
                center: 1.0,
                pairs: Vec::new(),
            };
        }

        let half_width = physical_radius.ceil() as usize;
        let sigma = f64::from(physical_radius) / 3.0;
        let weights: Vec<_> = (1..=half_width)
            .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
            .collect();
        let total = 1.0 + 2.0 * weights.iter().sum::<f64>();
        let pairs = weights
            .chunks(2)
            .enumerate()
            .filter_map(|(index, weights)| {
                let first = weights[0];
                let second = weights.get(1).copied().unwrap_or(0.0);
                let weight = first + second;

                // Subpixel radii can underflow the entire tail to zero.
                (weight > 0.0).then(|| {
                    [
                        ((2 * index + 1) as f64 + second / weight) as f32,
                        (weight / total) as f32,
                    ]
                })
            })
            .collect();

        Self {
            center: (1.0 / total) as f32,
            pairs,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    geometry: [f32; 4],
    // Axis, center weight, and paired positive-side sample count.
    parameters: [f32; 4],
}

impl BlurParams {
    fn new(context: &effect::Context, axis: BlurAxis, center: f32, pair_count: u32) -> Self {
        Self {
            geometry: geometry(context),
            parameters: [
                if axis == BlurAxis::Horizontal {
                    1.0
                } else {
                    0.0
                },
                if axis == BlurAxis::Vertical { 1.0 } else { 0.0 },
                center,
                pair_count as f32,
            ],
        }
    }
}

impl effect::LayerEffect for GaussianBlur {
    fn plan(&self, plan: &mut Plan<'_, Self>) {
        if self.normalized().radius > 0.0 {
            plan.push(BlurPass(BlurAxis::Horizontal));
            plan.push(BlurPass(BlurAxis::Vertical));
        }
    }

    fn expansion(&self) -> Padding {
        Padding::new(self.padding())
    }

    fn record_inputs(&self, inputs: &mut effect::LayerInputRecords) {
        inputs.record(&self.normalized());
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
        queue: &wgpu::Queue,
        _scratch: &mut effect::ScratchAllocator<'_>,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Prepared {
        let pipeline = pipelines.get_or_init::<BlurPipeline>();
        pipeline.prepare(
            device,
            queue,
            "iced_widget.isolated_layer.blur_gaussian",
            context,
            self.0,
            *effect,
            views.previous,
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
        let pipeline = pipelines
            .get::<BlurPipeline>()
            .expect("Gaussian blur pipeline");
        pipeline.0.render(
            encoder,
            views.output,
            context.physical_size,
            "iced_widget.isolated_layer.blur_gaussian",
            prepared,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Blur, DropShadow};
    use super::*;
    use crate::core::{Color, Vector};
    use effect::{EffectStack, LayerEffect};

    fn unpack(radius: f32) -> Vec<f64> {
        let kernel = GaussianKernel::new(radius);
        let mut weights = vec![0.0; radius.ceil() as usize + 1];
        weights[0] = f64::from(kernel.center);
        for [offset, weight] in kernel.pairs {
            assert!(offset.is_finite() && weight.is_finite());
            let index = offset.floor() as usize;
            weights[index] += f64::from(weight) * (1.0 - f64::from(offset.fract()));
            if offset.fract() > 0.0 {
                weights[index + 1] += f64::from(weight) * f64::from(offset.fract());
            }
        }
        weights
    }

    #[test]
    fn public_fields_normalize_like_the_constructor_including_cache_evidence() {
        assert_eq!(GaussianBlur::default(), GaussianBlur::new(0.0));
        for (radius, expected) in [
            (f32::NAN, 0.0f32),
            (f32::NEG_INFINITY, 0.0),
            (-10.0, 0.0),
            (-0.0, 0.0),
            (0.25, 0.25),
            (128.0, 128.0),
            (129.0, 128.0),
            (f32::INFINITY, 128.0),
        ] {
            let literal = GaussianBlur { radius };
            let canonical = GaussianBlur::new(radius);
            assert_eq!(canonical.radius.to_bits(), expected.to_bits());
            assert_eq!(literal.normalized(), canonical);
            assert_eq!(literal.expansion(), Padding::new(expected.ceil()));
            assert_eq!(
                EffectStack::from_effect(literal).input_evidence(),
                EffectStack::from_effect(canonical).input_evidence(),
            );

            let shadow = |blur| DropShadow {
                color: Color::BLACK,
                offset: Vector::new(-2.0, 3.0),
                blur: Blur::Gaussian(blur),
            };
            assert_eq!(
                shadow(literal).expansion(),
                Padding {
                    top: expected.ceil(),
                    right: expected.ceil(),
                    bottom: expected.ceil() + 3.0,
                    left: expected.ceil() + 2.0,
                }
            );
            assert_eq!(
                EffectStack::from_effect(shadow(literal)).input_evidence(),
                EffectStack::from_effect(shadow(canonical)).input_evidence(),
            );
        }
    }

    #[test]
    fn radius_changes_invalidate_standalone_and_shadow_caches() {
        let original = GaussianBlur::new(3.0);
        for changed in [GaussianBlur::new(3.01), GaussianBlur::new(0.0)] {
            assert_ne!(
                EffectStack::from_effect(original).input_evidence(),
                EffectStack::from_effect(changed).input_evidence(),
            );
            let shadow = |blur| DropShadow {
                blur: Blur::Gaussian(blur),
                ..DropShadow::default()
            };
            assert_ne!(
                EffectStack::from_effect(shadow(original)).input_evidence(),
                EffectStack::from_effect(shadow(changed)).input_evidence(),
            );
        }
    }

    #[test]
    fn paired_taps_reconstruct_dense_gaussian_weights() {
        for radius in [
            0.0,
            f32::MIN_POSITIVE,
            0.01,
            0.25,
            0.5,
            1.0,
            2.999,
            3.001,
            7.5,
            30.0,
            60.0,
            128.0,
            384.0,
        ] {
            let actual = unpack(radius);
            let reference: Vec<_> = (0..actual.len())
                .map(|i| {
                    if i == 0 {
                        1.0
                    } else {
                        (-0.5 * (3.0 * i as f64 / f64::from(radius)).powi(2)).exp()
                    }
                })
                .collect();
            let total = reference[0] + 2.0 * reference[1..].iter().sum::<f64>();
            for (actual, expected) in actual.iter().zip(&reference) {
                assert!(
                    (*actual - expected / total).abs() < 2.0e-7,
                    "radius {radius}: {actual} != {}",
                    expected / total
                );
            }
            assert!((actual[0] + 2.0 * actual[1..].iter().sum::<f64>() - 1.0).abs() < 1.0e-6);
            assert!(actual.windows(2).all(|pair| pair[0] >= pair[1]));
        }
        assert_eq!(GaussianKernel::new(30.0).pairs.len(), 15);
        assert_eq!(GaussianKernel::new(60.0).pairs.len(), 30);
        assert_eq!(GaussianKernel::new(31.0).pairs.len(), 16);
    }

    #[test]
    fn broad_impulse_has_no_holes_and_kernel_growth_has_small_transitions() {
        let weights = unpack(30.0);
        assert_eq!(weights.len(), 31);
        assert!(weights.iter().all(|weight| *weight > 0.0));
        assert!(weights[0] < 0.041); // The former sparse kernel retained 0.299.
        for radius in [1.0, 2.0, 3.0, 7.0, 30.0, 60.0, 128.0] {
            let mut before = unpack(radius - 0.0001);
            let after = unpack(radius + 0.0001);
            before.resize(after.len(), 0.0);
            let difference: f64 = before
                .iter()
                .zip(&after)
                .enumerate()
                .map(|(i, (a, b))| (a - b).abs() * if i == 0 { 1.0 } else { 2.0 })
                .sum();
            assert!(difference < 0.003, "radius {radius}: {difference}");
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    #[ignore = "requires a native GPU; run explicitly with --ignored"]
    fn gaussian_kernel_cache_reuses_coefficients_and_bounds_animation_growth() {
        use futures::executor::block_on;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter");
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("GPU device");
        let mut pipeline = <BlurPipeline as effect::Pipeline>::new(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let first = pipeline.kernel(&device, &queue, 30.0);
        let same = pipeline.kernel(&device, &queue, 30.0);
        assert!(Arc::ptr_eq(&first, &same));
        let scaled = pipeline.kernel(&device, &queue, 60.0);
        assert!(!Arc::ptr_eq(&first, &scaled));
        assert_eq!(scaled.pair_count, 2 * first.pair_count);
        for radius in 1..=32 {
            let _ = pipeline.kernel(&device, &queue, radius as f32);
        }
        assert_eq!(pipeline.1.len(), BlurPipeline::CACHED_KERNELS);
        assert_eq!(first.pair_count, 15); // Evicted kernels remain valid for prepared passes.
        let _ = device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU completion");
    }
}
