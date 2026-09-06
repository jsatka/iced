//! Application-scoped, renderer-pooled supplementary effect textures.

use super::{Diagnostics, Pool, Target};
use crate::core::Size;

/// An invalid scratch texture request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ScratchError {
    /// Both dimensions must be nonzero.
    #[error("scratch texture dimensions must be nonzero")]
    Empty,
    /// The requested extent exceeds the device's 2D texture limit.
    #[error("scratch texture exceeds the device's 2D texture limit")]
    TooLarge,
}

/// A supplementary texture valid for one prepared pass application.
///
/// Keep this handle and its bindings in the pass's prepared state, never in
/// shared pipeline storage. Dropping a handle does not release the renderer's
/// lease early. Every sampled region must be initialized by the pass.
///
/// Textures use the effect format, one mip, one sample, and render-attachment
/// and texture-binding usages, plus copy-source and copy-destination usages for
/// reuse with layer targets. Their initial contents are unspecified.
#[derive(Debug)]
pub struct ScratchTexture {
    view: wgpu::TextureView,
    physical_size: Size<u32>,
    backing_extent: Size<u32>,
}

impl ScratchTexture {
    /// Returns the texture view for sampling or rendering.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Returns the requested valid physical size.
    pub fn physical_size(&self) -> Size<u32> {
        self.physical_size
    }

    /// Returns the full pooled allocation extent.
    pub fn backing_extent(&self) -> Size<u32> {
        self.backing_extent
    }

    /// Returns the valid normalized texture-coordinate maximum.
    pub fn valid_uv(&self) -> [f32; 2] {
        [
            self.physical_size.width as f32 / self.backing_extent.width as f32,
            self.physical_size.height as f32 / self.backing_extent.height as f32,
        ]
    }
}

/// Allocates supplementary textures for the current pass application.
///
/// Leases remain exclusive until the renderer drops the application's prepared
/// bindings after encoding. No manual release or allocation count is required.
/// Free textures participate in the renderer's end-of-frame budget; active draw
/// memory and independently allocated private resources are not budget capped.
pub struct ScratchAllocator<'a> {
    device: &'a wgpu::Device,
    format: wgpu::TextureFormat,
    pool: &'a mut Pool,
    diagnostics: &'a mut Diagnostics,
    leases: Vec<Target>,
}

impl<'a> ScratchAllocator<'a> {
    pub(crate) fn new(
        device: &'a wgpu::Device,
        format: wgpu::TextureFormat,
        pool: &'a mut Pool,
        diagnostics: &'a mut Diagnostics,
    ) -> Self {
        Self {
            device,
            format,
            pool,
            diagnostics,
            leases: Vec::new(),
        }
    }

    /// Leases a texture of the requested valid size.
    ///
    /// This validates dimensions, not device loss or out-of-memory conditions.
    pub fn allocate(&mut self, size: Size<u32>) -> Result<ScratchTexture, ScratchError> {
        let (target, hit) = self.pool.lease_scratch(self.device, self.format, size)?;
        self.diagnostics.record_allocation(target.byte_size(), hit);
        let handle = ScratchTexture {
            view: target.view.clone(),
            physical_size: size,
            backing_extent: target.extent,
        };
        self.leases.push(target);
        Ok(handle)
    }

    pub(crate) fn finish(mut self) -> Vec<Target> {
        std::mem::take(&mut self.leases)
    }
}

impl Drop for ScratchAllocator<'_> {
    fn drop(&mut self) {
        // A prepare panic must not strand the allocator's leases. Successful
        // preparation transfers them to PreparedEffectPass with finish().
        release(self.pool, std::mem::take(&mut self.leases));
    }
}

/// Returns renderer-owned scratch leases after their prepared bindings are dropped.
/// Also used by the allocator's unwind guard.
pub(crate) fn release(pool: &mut Pool, targets: Vec<Target>) {
    for target in targets {
        pool.release(target);
    }
}

pub(super) fn backing_extent(size: Size<u32>, maximum: u32) -> Result<Size<u32>, ScratchError> {
    fn axis(value: u32, maximum: u32) -> Result<u32, ScratchError> {
        if value == 0 {
            return Err(ScratchError::Empty);
        }
        if value > maximum {
            return Err(ScratchError::TooLarge);
        }
        let rounded = if value <= 64 {
            value.next_power_of_two()
        } else {
            value.div_ceil(64).checked_mul(64).unwrap_or(maximum)
        };
        Ok(rounded.min(maximum))
    }
    Ok(Size::new(
        axis(size.width, maximum)?,
        axis(size.height, maximum)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_levels_do_not_allocate_full_layer_size_classes() {
        for (requested, expected) in [(1, 1), (3, 4), (17, 32), (64, 64), (65, 128)] {
            assert_eq!(
                backing_extent(Size::new(requested, 1), 4096),
                Ok(Size::new(expected, 1))
            );
        }
    }

    #[test]
    fn dimensions_are_checked_before_rounding() {
        assert_eq!(
            backing_extent(Size::new(0, 1), 4096),
            Err(ScratchError::Empty)
        );
        assert_eq!(
            backing_extent(Size::new(4097, 1), 4096),
            Err(ScratchError::TooLarge)
        );
        assert_eq!(backing_extent(Size::new(65, 1), 100), Ok(Size::new(100, 1)));
        assert_eq!(
            backing_extent(Size::new(u32::MAX, 1), u32::MAX),
            Ok(Size::new(u32::MAX, 1))
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    #[ignore = "requires a native GPU; run explicitly with --ignored"]
    fn leases_survive_early_handle_drops_and_unwind_releases_allocations() {
        use futures::executor::block_on;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("native GPU adapter");
        let (device, _) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("native GPU device");
        let mut pool = Pool::default();
        let mut diagnostics = Diagnostics::default();
        let mut allocator = ScratchAllocator::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            &mut pool,
            &mut diagnostics,
        );
        assert!(matches!(
            allocator.allocate(Size::new(0, 1)),
            Err(ScratchError::Empty)
        ));
        let first = allocator.allocate(Size::new(7, 5)).expect("first lease");
        let view = first.view().clone();
        drop(first);
        let second = allocator.allocate(Size::new(7, 5)).expect("second lease");
        assert_ne!(view.texture(), second.view().texture());
        let leases = allocator.finish();
        assert_eq!(diagnostics.allocated_bytes, 2 * 8 * 8 * 4);
        drop(view);
        drop(second);
        release(&mut pool, leases);
        assert_eq!(diagnostics.allocated_bytes, 2 * 8 * 8 * 4);
        let bytes = pool.bytes();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut allocator = ScratchAllocator::new(
                &device,
                wgpu::TextureFormat::Rgba8Unorm,
                &mut pool,
                &mut diagnostics,
            );
            let _texture = allocator.allocate(Size::new(7, 5)).expect("reused lease");
            panic!("intentional preparation failure");
        }));
        assert!(result.is_err());
        assert_eq!(pool.bytes(), bytes);
        assert_eq!(diagnostics.allocated_bytes, 2 * 8 * 8 * 4);
        assert_eq!(pool.trim_to_bytes(bytes, bytes).total, 0);
        assert_eq!(pool.trim_to_bytes(0, bytes).total, 2);
        assert_eq!(pool.bytes(), 0);
    }
}
