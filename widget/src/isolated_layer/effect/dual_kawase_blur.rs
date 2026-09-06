use super::canonical;
use super::pipeline::{Prepared, TexturePipeline, sampler_entry, texture_entry, uniform_entry};
use crate::core::{Color, Padding, Size, Vector};
use crate::renderer::wgpu::isolated_layer::effect::{
    self, PipelineRegistry, Plan, Requirements, ScratchAllocator, ScratchTexture, TextureViews,
};
use crate::renderer::wgpu::shader::isolated_layer::effect as shader;
use crate::renderer::wgpu::wgpu;

use bytemuck::{Pod, Zeroable};

/// Dual Kawase blur settings and effect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualKawaseBlur {
    radius: f32,
    depth: u32,
}

impl DualKawaseBlur {
    const DEFAULT_DEPTH: u32 = 3;

    /// Creates new blur settings with given radius and default downsample levels.
    pub fn new(radius: f32) -> Self {
        Self {
            radius: canonical(radius, 0.0, 128.0),
            depth: Self::DEFAULT_DEPTH,
        }
    }

    /// Sets the requested number of downsample levels, clamped to 1–8.
    /// Actual depth may be smaller when neither texture dimension can shrink.
    ///
    /// Depth `d` uses `2 * d` draws and `2 * d - 1` supplementary textures.
    pub fn pyramid_depth(mut self, depth: u32) -> Self {
        self.depth = depth.clamp(1, 8);
        self
    }

    /// Returns the approximate logical-pixel radius.
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Returns the requested pyramid depth.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Practical logical-pixel estimate for required effect padding.
    ///
    /// Extreme zoom/depth combinations may clip faint tails.
    pub(super) fn padding(self) -> f32 {
        if self.radius == 0.0 {
            0.0
        } else {
            // Practical guard band, not an all-DPI mathematical support bound.
            (4.0 * self.radius + 6.0 * ((1 << self.depth) - 1) as f32).ceil()
        }
    }
}

impl Default for DualKawaseBlur {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl From<f32> for DualKawaseBlur {
    fn from(radius: f32) -> Self {
        Self::new(radius)
    }
}

impl effect::LayerEffect for DualKawaseBlur {
    fn plan(&self, plan: &mut Plan<'_, Self>) {
        if self.radius > 0.0 {
            plan.push(KawasePass);
        }
    }

    fn expansion(&self) -> Padding {
        Padding::new(self.padding())
    }

    fn is_translation_invariant(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KawasePass;

impl effect::Pass<DualKawaseBlur> for KawasePass {
    type Prepared = Chain;

    fn requirements(&self, _effect: &DualKawaseBlur) -> Requirements {
        Requirements::new().writes_every_pixel()
    }

    fn prepare(
        &self,
        effect: &DualKawaseBlur,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        scratch: &mut ScratchAllocator<'_>,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Chain {
        Chain::prepare(*effect, None, pipelines, device, scratch, context, views)
    }

    fn encode(
        &self,
        _effect: &DualKawaseBlur,
        pipelines: &PipelineRegistry<'_>,
        prepared: &Chain,
        encoder: &mut wgpu::CommandEncoder,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) {
        prepared.encode(pipelines, encoder, context, views.output);
    }
}

struct KawasePipeline(TexturePipeline);

impl effect::Pipeline for KawasePipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self(TexturePipeline::new(
            device,
            format,
            "iced_widget.isolated_layer.dual_kawase",
            shader::DUAL_KAWASE,
            &[
                texture_entry(0),
                texture_entry(1),
                sampler_entry(2),
                uniform_entry::<Params>(3),
            ],
        ))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    source: [f32; 4],
    original: [f32; 4],
    // Per-axis source-texel offsets, blend weight, and operation mode.
    sampling: [f32; 4],
    // Full-size output extent and physical shadow displacement.
    output: [f32; 4],
    color: [f32; 4],
}

fn geometry(size: Size<u32>, backing: Size<u32>) -> [f32; 4] {
    [
        size.width as f32,
        size.height as f32,
        backing.width as f32,
        backing.height as f32,
    ]
}

struct Step {
    output: Option<usize>,
    prepared: Prepared,
}

pub(super) struct Chain {
    // Drop bindings before the application handles (and renderer-owned leases).
    steps: Vec<Step>,
    textures: Vec<ScratchTexture>,
}

impl Chain {
    pub(super) fn prepare(
        settings: DualKawaseBlur,
        shadow: Option<(Color, Vector)>,
        pipelines: &mut PipelineRegistry<'_>,
        device: &wgpu::Device,
        scratch: &mut ScratchAllocator<'_>,
        context: &effect::Context,
        views: TextureViews<'_>,
    ) -> Self {
        let sizes = if settings.radius == 0.0 {
            Vec::new()
        } else {
            level_sizes(context.physical_size, settings.depth)
        };
        let sampling = Sampling::new(
            context.physical_size,
            &sizes,
            settings.radius * context.scale_factor,
        );
        let textures: Vec<_> = sizes
            .iter()
            // Preserve downsampled inputs for smooth level contributions.
            // Separate ascent destinations avoid sampling an attachment.
            .chain(sizes[..sizes.len().saturating_sub(1)].iter().rev())
            .map(|&size| {
                scratch
                    .allocate(size)
                    .expect("Kawase levels fit the validated layer target")
            })
            .collect();
        let pipeline = &pipelines.get_or_init::<KawasePipeline>().0;
        let original = geometry(context.physical_size, context.backing_extent);
        let mut steps = Vec::with_capacity(sizes.len() * 2 + usize::from(sizes.is_empty()));
        let (color, offset) = shadow.unwrap_or((Color::TRANSPARENT, Vector::ZERO));
        let mut params = Params {
            source: original,
            original,
            sampling: [
                sampling.offset[0],
                sampling.offset[1],
                sampling.weights[0],
                0.0,
            ],
            output: [
                original[0],
                original[1],
                offset.x * context.scale_factor,
                offset.y * context.scale_factor,
            ],
            color: crate::graphics::color::pack(color).components(),
        };
        let bind = |params: &Params, source, original| {
            pipeline.prepare(
                device,
                "iced_widget.isolated_layer.dual_kawase",
                params,
                &[(0, source), (1, original)],
                2,
                3,
            )
        };
        for (index, _) in sizes.iter().enumerate() {
            let source = if index == 0 {
                params.source = original;
                views.stage_input
            } else {
                let texture = &textures[index - 1];
                params.source = geometry(texture.physical_size(), texture.backing_extent());
                texture.view()
            };
            steps.push(Step {
                output: Some(index),
                prepared: bind(&params, source, views.stage_input),
            });
        }
        let mut previous = sizes.len().saturating_sub(1);
        for index in (0..sizes.len()).rev() {
            let texture = &textures[previous];
            params.source = geometry(texture.physical_size(), texture.backing_extent());
            params.sampling[2] = sampling.weights[index];
            let original_view = if index == 0 {
                params.original = original;
                views.stage_input
            } else {
                let original = &textures[index - 1];
                params.original = geometry(original.physical_size(), original.backing_extent());
                original.view()
            };
            params.sampling[3] = if index != 0 {
                1.0
            } else if shadow.is_some() {
                3.0
            } else {
                2.0
            };
            let output = (index > 0).then_some(textures.len() - index);
            steps.push(Step {
                output,
                prepared: bind(&params, texture.view(), original_view),
            });
            if let Some(output) = output {
                previous = output;
            }
        }
        if sizes.is_empty() {
            params.sampling[3] = if shadow.is_some() { 3.0 } else { 2.0 };
            steps.push(Step {
                output: None,
                prepared: bind(&params, views.stage_input, views.stage_input),
            });
        }
        Self { steps, textures }
    }

    pub(super) fn encode(
        &self,
        pipelines: &PipelineRegistry<'_>,
        encoder: &mut wgpu::CommandEncoder,
        context: &effect::Context,
        output: &wgpu::TextureView,
    ) {
        let pipeline = &pipelines
            .get::<KawasePipeline>()
            .expect("Kawase pipeline")
            .0;
        for step in &self.steps {
            let (view, size) = step
                .output
                .map_or((output, context.physical_size), |index| {
                    let texture = &self.textures[index];
                    (texture.view(), texture.physical_size())
                });
            pipeline.render(
                encoder,
                view,
                size,
                "iced_widget.isolated_layer.dual_kawase",
                &step.prepared,
            );
        }
    }
}

pub(super) fn level_sizes(mut size: Size<u32>, depth: u32) -> Vec<Size<u32>> {
    let mut levels = Vec::with_capacity(depth as usize);
    for _ in 0..depth {
        let next = Size::new(
            size.width.div_ceil(2).max(1),
            size.height.div_ceil(2).max(1),
        );
        if next == size {
            break;
        }
        levels.push(next);
        size = next;
    }
    levels
}

struct Sampling {
    offset: [f32; 2],
    // Contribution of the reconstructed next level at each destination level.
    weights: Vec<f32>,
}

impl Sampling {
    fn new(full: Size<u32>, levels: &[Size<u32>], radius: f32) -> Self {
        // Phase-averaged kernel variance plus a bilinear reconstruction term.
        // Exact at offset 2 for dyadic interior impulses; approximate for odd
        // dimensions and other offsets. Work is bounded by the requested depth.
        let mut base = [0.0; 2];
        let mut coefficient = [0.0; 2];
        let mut previous = full;
        let mut minimum = vec![0.0];
        for level in levels {
            for (axis, (full, before, after)) in [
                (full.width, previous.width, level.width),
                (full.height, previous.height, level.height),
            ]
            .into_iter()
            .enumerate()
            {
                let down = (full as f32 / before as f32).powi(2);
                let up = (full as f32 / after as f32).powi(2);
                base[axis] += 0.25 * down + 0.1875 * up;
                coefficient[axis] += 0.125 * down + up / 3.0;
            }
            previous = *level;
            minimum.push((base[0] + 4.0 * coefficient[0]).max(base[1] + 4.0 * coefficient[1]));
        }
        if levels.is_empty() {
            base = [0.1875; 2];
            coefficient = [1.0 / 3.0; 2];
            minimum.push(base[0] + 4.0 * coefficient[0]);
        }
        // Bound arithmetic even for extreme, otherwise valid renderer scales.
        let variance = radius.min(1.0e10).powi(2);
        let deepest = *minimum.last().expect("at least one filtering interval");
        Self {
            offset: if variance <= deepest {
                [2.0; 2]
            } else {
                std::array::from_fn(|axis| (4.0 + (variance - deepest) / coefficient[axis]).sqrt())
            },
            weights: minimum
                .windows(2)
                .map(|range| ((variance - range[0]) / (range[1] - range[0])).clamp(0.0, 1.0))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_canonical_and_depth_is_independent_of_radius() {
        assert_eq!(DualKawaseBlur::default(), DualKawaseBlur::new(0.0));
        for radius in [f32::NAN, f32::NEG_INFINITY, -1.0, -0.0] {
            assert_eq!(
                DualKawaseBlur::new(radius).radius().to_bits(),
                0.0f32.to_bits()
            );
        }
        assert_eq!(DualKawaseBlur::new(f32::INFINITY).radius(), 128.0);
        assert_eq!(DualKawaseBlur::new(1.0).pyramid_depth(0).depth(), 1);
        assert_eq!(DualKawaseBlur::new(1.0).pyramid_depth(u32::MAX).depth(), 8);
        for radius in [0.0, 0.01, 1.0, 128.0] {
            assert_eq!(DualKawaseBlur::new(radius).depth(), 3);
        }
    }

    #[test]
    fn dimensions_are_ceil_halved_and_stop_at_one() {
        assert_eq!(
            level_sizes(Size::new(19, 11), 8),
            [
                Size::new(10, 6),
                Size::new(5, 3),
                Size::new(3, 2),
                Size::new(2, 1),
                Size::new(1, 1),
            ]
        );
        assert!(level_sizes(Size::new(1, 1), 8).is_empty());
    }

    #[test]
    fn blend_is_continuous_and_zero_is_exact() {
        let full = Size::new(1024, 1024);
        for depth in 1..=8 {
            let levels = level_sizes(full, depth);
            assert_eq!(Sampling::new(full, &levels, 0.0).weights[0], 0.0);
            let low = Sampling::new(full, &levels, 0.001);
            assert!(low.weights[0] > 0.0 && low.weights[0] < 0.00001);
            let mut previous = 0.0;
            for radius in [0.01, 0.1, 1.0, 4.0, 12.0, 32.0, 128.0] {
                let sampling = Sampling::new(full, &levels, radius);
                assert!(sampling.weights[0] >= previous && sampling.weights[0] <= 1.0);
                assert!(
                    sampling
                        .offset
                        .iter()
                        .all(|value| value.is_finite() && *value >= 2.0)
                );
                previous = sampling.weights[0];
            }
        }
        assert_eq!(DualKawaseBlur::new(0.0).padding(), 0.0);
    }

    #[test]
    fn level_contributions_are_monotonic_and_ordered() {
        let full = Size::new(151, 117);
        let levels = level_sizes(full, 8);
        let mut previous = vec![0.0; levels.len()];
        for radius in (0..=1280).map(|value| value as f32 / 10.0) {
            let sampling = Sampling::new(full, &levels, radius);
            for (weight, previous) in sampling.weights.iter().zip(&mut previous) {
                assert!(*weight >= *previous);
                *previous = *weight;
            }
            assert!(sampling.weights.windows(2).all(|pair| pair[0] >= pair[1]));
            assert!(
                sampling
                    .weights
                    .iter()
                    .filter(|&&weight| weight > 0.0 && weight < 1.0)
                    .count()
                    <= 1
            );
        }
    }

    #[test]
    fn offset_transition_is_continuous_for_unequal_axes() {
        for full in [Size::new(1024, 1024), Size::new(151, 117), Size::new(1, 17)] {
            for depth in 1..=8 {
                let levels = level_sizes(full, depth);
                let (mut low, mut high) = (0.0, 1024.0);
                for _ in 0..32 {
                    let middle = (low + high) * 0.5;
                    if Sampling::new(full, &levels, middle).weights.last() == Some(&1.0) {
                        high = middle;
                    } else {
                        low = middle;
                    }
                }
                let before = Sampling::new(full, &levels, low - 0.0001);
                let after = Sampling::new(full, &levels, high + 0.0001);
                for (a, b) in before.offset.into_iter().zip(after.offset) {
                    assert!((a - b).abs() < 0.005, "{full:?} depth={depth}: {a} → {b}");
                }
            }
        }
    }
}
