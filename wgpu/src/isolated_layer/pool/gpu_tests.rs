//! Explicit native checks for role changes in the shared texture pool.

use super::*;
use crate::core::isolated_layer::{
    Composite, CompositePositioning, ContentChangeHandle, Layer, SurfaceHandle,
};
use crate::core::text::Renderer as _;
use crate::core::{Color, Font, Point, Rectangle, Renderer as _, renderer};
use crate::graphics::{Shell, Viewport};
use crate::isolated_layer::{self, Renderer as _, ScratchAllocator};
use crate::{Engine, Renderer};
use futures::executor::block_on;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const EXTENT: Size<u32> = Size::new(64, 64);
const BYTES: u64 = 64 * 64 * 4;

fn engine() -> Engine {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("native GPU adapter");
    eprintln!("GPU: {:?}", adapter.get_info());
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("native GPU device");
    Engine::new(&adapter, device, queue, FORMAT, None, Shell::headless())
}

fn viewport(size: Size<u32>) -> Viewport {
    Viewport::with_physical_size(size, renderer::Scale::default())
}

/// Keep every lease active until all requests have completed, then poison the
/// entire backing (including padding) before making it available to the renderer.
fn seed_scratch(renderer: &mut Renderer, count: usize) -> Vec<wgpu::Texture> {
    let state = &mut renderer.isolated_layers;
    let mut allocator = ScratchAllocator::new(
        &renderer.engine.device,
        FORMAT,
        &mut state.pool,
        &mut state.diagnostics,
    );
    let handles: Vec<_> = (0..count)
        .map(|_| allocator.allocate(EXTENT).unwrap())
        .collect();
    let textures = handles
        .iter()
        .map(|handle| handle.view().texture().clone())
        .collect();
    let leases = allocator.finish();
    for target in &leases {
        renderer.engine.queue.write_texture(
            target.texture.as_image_copy(),
            &[255, 0, 255, 255].repeat(64 * 64),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
            target.texture.size(),
        );
    }
    drop(handles);
    scratch::release(&mut state.pool, leases);
    textures
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn role_changes_reuse_exact_backings_and_account_for_the_current_request() {
    let engine = engine();
    let mut state = isolated_layer::State::default();
    state.begin_frame();
    let (target, hit) = state
        .pool
        .lease_layer(&engine.device, FORMAT, Size::new(45, 47));
    assert!(!hit);
    assert!(target.text_viewport.is_none());
    let texture = target.texture.clone();
    let address = std::ptr::from_ref::<TargetStorage>(&target);
    state.pool.release(target);

    for _ in 0..4 {
        let mut allocator = ScratchAllocator::new(
            &engine.device,
            FORMAT,
            &mut state.pool,
            &mut state.diagnostics,
        );
        let handle = allocator.allocate(Size::new(49, 61)).unwrap();
        assert_eq!(handle.view().texture(), &texture);
        assert_eq!(handle.physical_size(), Size::new(49, 61));
        assert_eq!(handle.backing_extent(), EXTENT);
        assert_eq!(handle.valid_uv(), [49.0 / 64.0, 61.0 / 64.0]);
        let leases = allocator.finish();
        assert_eq!(std::ptr::from_ref::<TargetStorage>(&leases[0]), address);
        assert_eq!(state.pool.bytes(), 0);
        drop(handle);
        scratch::release(&mut state.pool, leases);
        assert_eq!(state.pool.scratch_bytes(), BYTES);

        let (target, hit) = state.pool.lease_layer(&engine.device, FORMAT, EXTENT);
        assert!(hit);
        assert_eq!(target.texture, texture);
        assert_eq!(std::ptr::from_ref::<TargetStorage>(&target), address);
        assert!(target.text_viewport.is_none());
        state.pool.release(target);
        assert_eq!(state.pool.scratch_bytes(), 0);
        assert_eq!(state.pool.bytes(), BYTES);
    }
    assert_eq!(state.diagnostics.allocated_bytes, 0);

    // Neither an incompatible size nor format may consume the existing backing.
    let (small, hit) = state
        .pool
        .lease_scratch(&engine.device, FORMAT, Size::new(7, 5))
        .unwrap();
    assert!(!hit);
    assert_eq!(small.extent, Size::new(8, 8));
    let (_wide, hit) = state
        .pool
        .lease_scratch(&engine.device, FORMAT, Size::new(65, 64))
        .unwrap();
    assert!(!hit);
    let (float, hit) = state
        .pool
        .lease_scratch(&engine.device, wgpu::TextureFormat::Rgba16Float, EXTENT)
        .unwrap();
    assert!(!hit);
    assert_eq!(float.byte_size(), 2 * BYTES);
    assert_eq!(state.pool.bytes(), BYTES);
    let (active, hit) = state
        .pool
        .lease_scratch(&engine.device, FORMAT, EXTENT)
        .unwrap();
    assert!(hit);
    let (other, hit) = state.pool.lease_layer(&engine.device, FORMAT, EXTENT);
    assert!(!hit);
    assert_ne!(active.texture, other.texture);
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn trimming_uses_release_order_and_one_shared_budget_without_idle_expiry() {
    let engine = engine();
    let mut state = isolated_layer::State::default();
    let (layer, _) = state.pool.lease_layer(&engine.device, FORMAT, EXTENT);
    let (scratch, _) = state
        .pool
        .lease_scratch(&engine.device, FORMAT, EXTENT)
        .unwrap();
    // Return scratch first in the same frame: role must not override recency.
    state.pool.release(scratch);
    state.pool.release(layer);
    state.set_limits(isolated_layer::Limits::new(2 * BYTES, 2));
    for frame in [1, 122, 10_000, u64::MAX, 0] {
        state.frame = frame;
        state.finish_frame();
        assert_eq!(state.diagnostics.retained_bytes(), 2 * BYTES);
    }
    state.set_limits(isolated_layer::Limits::new(BYTES, 2));
    state.finish_frame();
    assert_eq!(state.pool.scratch_bytes(), 0);
    assert_eq!(state.diagnostics.retained_bytes(), BYTES);

    let (oversized, _) = state
        .pool
        .lease_layer(&engine.device, FORMAT, Size::new(128, 128));
    state.pool.release(oversized);
    state.finish_frame();
    assert_eq!(state.pool.free_targets().next().unwrap().extent, EXTENT);
    assert_eq!(state.diagnostics.retained_bytes(), BYTES);
    state.set_limits(isolated_layer::Limits::new(0, 2));
    state.finish_frame();
    assert_eq!(state.diagnostics.retained_bytes(), 0);
}

fn retain_output(
    state: &mut isolated_layer::State,
    device: &wgpu::Device,
    priority: crate::core::isolated_layer::CacheResidencyPriority,
) -> crate::core::isolated_layer::CacheRequest {
    let surface = SurfaceHandle::new();
    let content = ContentChangeHandle::new();
    let request = surface.cache_request_with(priority, [&content]);
    let context = isolated_layer::Context::root(&viewport(EXTENT), FORMAT);
    let key = isolated_layer::OutputKey::new(
        &request,
        isolated_layer::LayerInputRecords::new().finish(),
        &context,
        Rectangle::with_size(Size::new(64.0, 64.0)),
        false,
        7,
    );
    let lease = state
        .registry
        .lease_output(&request, key.clone(), state.frame);
    let (target, _) = state.pool.lease_layer(device, FORMAT, EXTENT);
    assert!(
        state
            .registry
            .store_output(lease.ticket.unwrap(), &request, key, state.frame, target)
            .stored()
    );
    request
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn frame_end_evicts_free_then_normal_then_protected_outputs() {
    use crate::core::isolated_layer::CacheResidencyPriority::{Normal, Protected};
    let engine = engine();
    let mut state = isolated_layer::State::default();
    state.begin_frame();
    let _ = retain_output(&mut state, &engine.device, Protected);
    let _ = retain_output(&mut state, &engine.device, Normal);
    let (free, _) = state.pool.lease_layer(&engine.device, FORMAT, EXTENT);
    state.pool.release(free);
    state.set_limits(isolated_layer::Limits::new(2 * BYTES, 2));
    state.finish_frame();
    assert_eq!(state.diagnostics.retained_bytes(), 2 * BYTES);
    assert_eq!(state.diagnostics.pool_bytes, 0);
    assert_eq!(state.diagnostics.normal_priority_bytes, BYTES);
    assert_eq!(state.diagnostics.protected_priority_bytes, BYTES);
    state.set_limits(isolated_layer::Limits::new(BYTES, 2));
    state.finish_frame();
    assert_eq!(state.diagnostics.retained_bytes(), BYTES);
    assert_eq!(state.diagnostics.protected_priority_bytes, BYTES);
    state.set_limits(isolated_layer::Limits::new(0, 2));
    state.finish_frame();
    assert_eq!(state.diagnostics.retained_bytes(), 0);
    assert_eq!(state.registry.metadata_counts().outputs, 2);
    assert_eq!(state.registry.metadata_counts().resident_outputs, 0);
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn active_replacement_excess_is_reported_and_trimmed_after_recovery() {
    let engine = engine();
    let mut state = isolated_layer::State::default();
    state.begin_frame();
    let request = retain_output(
        &mut state,
        &engine.device,
        crate::core::isolated_layer::CacheResidencyPriority::Protected,
    );
    let context = isolated_layer::Context::root(&viewport(EXTENT), FORMAT);
    let mut inputs = isolated_layer::LayerInputRecords::new();
    inputs.record(&1_u8);
    let key = isolated_layer::OutputKey::new(
        &request,
        inputs.finish(),
        &context,
        Rectangle::with_size(Size::new(64.0, 64.0)),
        false,
        7,
    );
    let active = state.registry.lease_output(&request, key, state.frame);
    assert!(!active.valid);
    assert!(active.ticket.is_some());
    state.set_limits(isolated_layer::Limits::new(0, 2));
    state.finish_frame();
    assert_eq!(state.registry.finish_frame(state.frame).output_leases, 1);
    assert_eq!(state.diagnostics.retained_bytes(), BYTES);
    state.begin_frame();
    state.finish_frame();
    assert_eq!(state.registry.finish_frame(state.frame).output_leases, 0);
    assert_eq!(state.diagnostics.retained_bytes(), 0);
}

fn draw_text(renderer: &mut Renderer, bounds: Rectangle) {
    renderer.fill_text(
        crate::core::Text {
            content: "Ag".into(),
            bounds: bounds.size(),
            size: 20.0.into(),
            line_height: crate::core::text::LineHeight::default(),
            font: Font::DEFAULT,
            align_x: crate::core::text::Alignment::Left,
            align_y: crate::core::alignment::Vertical::Top,
            shaping: crate::core::text::Shaping::Basic,
            wrapping: crate::core::text::Wrapping::None,
            ellipsis: crate::core::text::Ellipsis::default(),
            hint_factor: None,
        },
        Point::new(bounds.x + 3.0, bounds.y + 2.0),
        Color::WHITE,
        bounds,
    );
}

fn capture_text(renderer: &mut Renderer, size: Size<u32>, layer: Layer) -> Vec<u8> {
    let viewport = viewport(size);
    let bounds = Rectangle::with_size(viewport.logical_size());
    renderer.reset(bounds);
    renderer.with_isolated_layer(layer, |renderer| draw_text(renderer, bounds));
    renderer.screenshot(&viewport, Color::TRANSPARENT)
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn scratch_backings_render_text_and_preserve_lazy_viewports_across_size_changes() {
    let engine = engine();
    let mut renderer = Renderer::new(engine.clone(), renderer::Settings::default());
    let mut identity = None;
    for size in [Size::new(45, 47), Size::new(61, 57), Size::new(45, 47)] {
        let seeded = seed_scratch(&mut renderer, 1);
        if let Some(identity) = &identity {
            assert_eq!(&seeded[0], identity);
            assert!(
                renderer
                    .isolated_layers
                    .pool
                    .free_targets()
                    .next()
                    .unwrap()
                    .text_viewport
                    .is_some()
            );
        } else {
            assert!(
                renderer
                    .isolated_layers
                    .pool
                    .free_targets()
                    .next()
                    .unwrap()
                    .text_viewport
                    .is_none()
            );
            identity = Some(seeded[0].clone());
        }
        let bounds = Rectangle::with_size(viewport(size).logical_size());
        let layer = Layer::new(bounds, bounds);
        let mut fresh = Renderer::new(engine.clone(), renderer::Settings::default());
        let expected = capture_text(&mut fresh, size, layer.clone());
        let actual = capture_text(&mut renderer, size, layer);
        assert!(
            actual.chunks_exact(4).any(|pixel| pixel[3] != 0),
            "text must be visible"
        );
        assert_eq!(actual, expected);
        assert_eq!(renderer.isolated_layers.diagnostics.allocated_bytes, 0);
        assert_eq!(renderer.isolated_layers.pool.scratch_bytes(), 0);
        assert_eq!(
            renderer
                .isolated_layers
                .pool
                .free_targets()
                .next()
                .unwrap()
                .texture,
            seeded[0]
        );
        assert!(
            renderer
                .isolated_layers
                .pool
                .free_targets()
                .next()
                .unwrap()
                .text_viewport
                .is_some()
        );
    }
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn budget_trimmed_backings_remain_valid_for_encoded_commands() {
    let engine = engine();
    let mut renderer = Renderer::new(engine.clone(), renderer::Settings::default());
    renderer.set_isolated_layer_limits(isolated_layer::Limits::new(0, 2));
    // Seed and poison a backing, then drop even these extra handles so only the
    // encoded commands can keep it alive after frame-end ownership trimming.
    drop(seed_scratch(&mut renderer, 1));
    let mut fresh = Renderer::new(engine, renderer::Settings::default());
    let bounds = Rectangle::with_size(viewport(EXTENT).logical_size());
    let layer = Layer::new(bounds, bounds);
    let expected = capture_text(&mut fresh, EXTENT, layer.clone());
    assert_eq!(capture_text(&mut renderer, EXTENT, layer), expected);
    assert_eq!(renderer.isolated_layers.diagnostics.retained_bytes(), 0);
}

#[derive(Debug, Clone, PartialEq)]
struct Backdrop;

impl isolated_layer::LayerEffect for Backdrop {
    fn plan(&self, plan: &mut isolated_layer::Plan<'_, Self>) {
        plan.push(Self);
    }
}

impl isolated_layer::Pass<Backdrop> for Backdrop {
    type Prepared = ();

    fn requirements(&self, _: &Backdrop) -> isolated_layer::Requirements {
        isolated_layer::Requirements::new().with_backdrop()
    }

    fn prepare(
        &self,
        _: &Backdrop,
        _: &mut isolated_layer::PipelineRegistry<'_>,
        _: &wgpu::Device,
        _: &wgpu::Queue,
        _: &mut ScratchAllocator<'_>,
        _: &isolated_layer::EffectContext,
        _: isolated_layer::TextureViews<'_>,
    ) {
    }

    fn encode(
        &self,
        _: &Backdrop,
        _: &isolated_layer::PipelineRegistry<'_>,
        _: &(),
        encoder: &mut wgpu::CommandEncoder,
        context: &isolated_layer::EffectContext,
        views: isolated_layer::TextureViews<'_>,
    ) {
        encoder.copy_texture_to_texture(
            views.backdrop.unwrap().texture().as_image_copy(),
            views.output.texture().as_image_copy(),
            wgpu::Extent3d {
                width: context.physical_size.width,
                height: context.physical_size.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn capture_backdrop(renderer: &mut Renderer, positioning: CompositePositioning) -> Vec<u8> {
    let viewport = viewport(EXTENT);
    let bounds = Rectangle::with_size(viewport.logical_size());
    renderer.reset(bounds);
    draw_text(renderer, bounds);
    let child = Rectangle::new(Point::new(5.25, 7.5), Size::new(43.0, 41.0));
    renderer.with_isolated_layer_effects(
        isolated_layer::Layer::new(child, bounds)
            .composite(Composite::default().with_positioning(positioning)),
        isolated_layer::EffectStack::from_effect(Backdrop),
        |_| {},
    );
    renderer.screenshot(&viewport, Color::TRANSPARENT)
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn scratch_backings_support_backdrop_copies_and_subpixel_sampling() {
    let engine = engine();
    for positioning in [
        CompositePositioning::Snapped,
        CompositePositioning::Subpixel,
    ] {
        let mut renderer = Renderer::new(engine.clone(), renderer::Settings::default());
        let seeded = seed_scratch(&mut renderer, 4);
        let mut fresh = Renderer::new(engine.clone(), renderer::Settings::default());
        let expected = capture_backdrop(&mut fresh, positioning);
        assert_eq!(
            fresh.isolated_layer_diagnostics().allocated_bytes,
            4 * BYTES
        );
        assert_eq!(capture_backdrop(&mut renderer, positioning), expected);
        let state = &renderer.isolated_layers;
        assert_eq!(state.diagnostics.uncached_renders, 1);
        assert_eq!(state.diagnostics.allocated_bytes, 0);
        assert_eq!(state.pool.free_targets().count(), 4);
        assert!(
            state
                .pool
                .free_targets()
                .all(|target| seeded.contains(&target.texture))
        );
        assert_eq!(
            state
                .pool
                .free_targets()
                .filter(|target| target.text_viewport.is_some())
                .count(),
            2
        );
        assert_eq!(state.pool.scratch_bytes(), 0);
    }
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn retained_output_excludes_scratch_reuse_until_swept() {
    let engine = engine();
    let mut renderer = Renderer::new(engine, renderer::Settings::default());
    let seeded = seed_scratch(&mut renderer, 1);
    let address = std::ptr::from_ref::<TargetStorage>(
        renderer.isolated_layers.pool.free_targets().next().unwrap(),
    );
    let size = Size::new(45, 47);
    let bounds = Rectangle::with_size(viewport(size).logical_size());
    let surface = SurfaceHandle::new();
    let content = ContentChangeHandle::new();
    let layer = Layer::new(bounds, bounds).cache_output(&surface, [&content]);
    let pixels = capture_text(&mut renderer, size, layer.clone());
    assert_eq!(renderer.isolated_layers.pool.bytes(), 0);
    assert_eq!(renderer.isolated_layers.registry.bytes(), BYTES);
    assert_eq!(capture_text(&mut renderer, size, layer), pixels);
    assert_eq!(renderer.isolated_layers.diagnostics.output_cache_hits, 1);
    assert_eq!(renderer.isolated_layers.diagnostics.allocated_bytes, 0);

    let (other, hit) = renderer
        .isolated_layers
        .pool
        .lease_scratch(&renderer.engine.device, FORMAT, EXTENT)
        .unwrap();
    assert!(!hit);
    assert_ne!(other.texture, seeded[0]);
    drop(other);
    renderer.set_isolated_layer_limits(isolated_layer::Limits::new(BYTES, 0));
    renderer.reset(bounds);
    let _ = renderer.screenshot(&viewport(size), Color::TRANSPARENT);
    assert_eq!(renderer.isolated_layers.registry.bytes(), 0);
    let (released, hit) = renderer
        .isolated_layers
        .pool
        .lease_scratch(&renderer.engine.device, FORMAT, EXTENT)
        .unwrap();
    assert!(hit);
    assert_eq!(released.texture, seeded[0]);
    assert_eq!(std::ptr::from_ref::<TargetStorage>(&released), address);
}

#[test]
#[ignore = "requires a native GPU; run explicitly with --ignored"]
fn draw_diagnostics_count_processed_layers_and_allocations_before_trim() {
    let mut renderer = Renderer::new(engine(), renderer::Settings::default());
    let size = Size::new(45, 47);
    let viewport = viewport(size);
    let bounds = Rectangle::with_size(viewport.logical_size());
    let surface = SurfaceHandle::new();
    let content = ContentChangeHandle::new();
    let parent = Layer::new(bounds, bounds).cache_output(&surface, [&content]);
    let child = Layer::new(bounds, bounds);

    for draw in 0..2 {
        renderer.reset(bounds);
        renderer.with_isolated_layer(parent.clone(), |renderer| {
            renderer.with_isolated_layer(child.clone(), |renderer| draw_text(renderer, bounds));
        });
        let _ = renderer.screenshot(&viewport, Color::TRANSPARENT);
        let stats = renderer.isolated_layer_diagnostics();
        if draw == 0 {
            assert_eq!(stats.output_cache_misses, 1);
            assert_eq!(stats.uncached_renders, 1);
            assert_eq!(stats.output_cache_hits, 0);
            assert_eq!(stats.allocated_bytes, 2 * BYTES);
            assert_eq!(stats.pool_bytes, BYTES);
            assert_eq!(stats.normal_priority_bytes, BYTES);
        } else {
            assert_eq!(stats.output_cache_hits, 1);
            assert_eq!(stats.layer_count(), 1);
            assert_eq!(stats.allocated_bytes, 0);
        }
    }

    // A fresh allocation remains counted even when a zero budget discards it.
    let mut renderer = Renderer::new(engine(), renderer::Settings::default());
    renderer.set_isolated_layer_limits(isolated_layer::Limits::new(0, 0));
    let _ = capture_text(&mut renderer, size, child);
    let stats = renderer.isolated_layer_diagnostics();
    assert_eq!(stats.uncached_renders, 1);
    assert_eq!(stats.allocated_bytes, BYTES);
    assert_eq!(stats.retained_bytes(), 0);

    renderer.reset(bounds);
    let _ = renderer.screenshot(&viewport, Color::TRANSPARENT);
    assert_eq!(
        renderer.isolated_layer_diagnostics(),
        isolated_layer::Diagnostics::default()
    );
}
