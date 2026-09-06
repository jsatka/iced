//! Renderer-owned texture targets and their reuse pool.

use super::scratch::{self, ScratchError};
use crate::core::Size;
use crate::text;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

#[cfg(test)]
mod reuse_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lease_benchmark;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_tests;

/// Physical-pixel increment used to quantize each pooled backing-texture dimension.
const SIZE_INCREMENT: u32 = 64;

/// A normalized two-dimensional backing-texture size.
///
/// Dimensions are stored in units of [`SIZE_INCREMENT`]. This normalization is
/// specific to layer requests; shared pool entries use their exact pixel extent.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SizeClass {
    width: u16,
    height: u16,
}

impl SizeClass {
    fn new(requested_size: Size<u32>, maximum_dimension: u32) -> Option<Self> {
        fn normalize_dimension(requested: u32, maximum: u32) -> Option<u16> {
            if requested == 0 {
                return None;
            }
            let req_units = requested.div_ceil(SIZE_INCREMENT);
            let max_units = maximum / SIZE_INCREMENT;
            if req_units > max_units {
                return None;
            }
            u16::try_from(req_units).ok()
        }

        Some(Self {
            width: normalize_dimension(requested_size.width, maximum_dimension)?,
            height: normalize_dimension(requested_size.height, maximum_dimension)?,
        })
    }

    fn extent(self) -> Size<u32> {
        Size::new(
            u32::from(self.width) * SIZE_INCREMENT,
            u32::from(self.height) * SIZE_INCREMENT,
        )
    }
}

/// An exclusive pointer-sized lease to a renderer-owned texture target.
pub(crate) type Target = Lease<TargetStorage>;

/// Moves exclusive ownership without moving the backing's CPU storage. A lease
/// cannot also be in the free pool; dropping it preserves ordinary RAII cleanup.
#[repr(transparent)]
pub(crate) struct Lease<T>(Box<T>);

impl<T> Deref for Lease<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Lease<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl Target {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        extent: Size<u32>,
        role: Role,
    ) -> Self {
        Self(Box::new(TargetStorage::new(device, format, extent, role)))
    }
}

/// Stable CPU storage for a texture, its view, and lazy text resources.
pub(crate) struct TargetStorage {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub extent: Size<u32>,
    text_viewport: Option<text::Viewport>,
    format: wgpu::TextureFormat,
    last_role: Role,
}

impl TargetStorage {
    pub fn byte_size(&self) -> u64 {
        texture_byte_size(self.format, self.extent)
    }

    /// Prepares text resources only for targets that will render child content.
    pub fn prepare_text_viewport(
        &mut self,
        device: &wgpu::Device,
        pipeline: &text::Pipeline,
        queue: &wgpu::Queue,
        size: Size<u32>,
    ) -> &mut text::Viewport {
        let viewport = self
            .text_viewport
            .get_or_insert_with(|| pipeline.create_viewport(device));
        viewport.update(queue, size);
        viewport
    }

    pub fn text_viewport(&self) -> &text::Viewport {
        self.text_viewport
            .as_ref()
            .expect("prepared capture text viewport")
    }

    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        extent: Size<u32>,
        role: Role,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_wgpu.isolated_layer.target"),
            size: wgpu::Extent3d {
                width: extent.width,
                height: extent.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            extent,
            text_viewport: None,
            format,
            last_role: role,
        }
    }
}

pub(super) fn texture_byte_size(format: wgpu::TextureFormat, extent: Size<u32>) -> u64 {
    let (block_width, block_height) = format.block_dimensions();
    let bytes_per_block = format
        .block_copy_size(None)
        .expect("GPU color targets must have a defined texel-block size");

    u64::from(extent.width.div_ceil(block_width))
        .saturating_mul(u64::from(extent.height.div_ceil(block_height)))
        .saturating_mul(u64::from(bytes_per_block))
}

/// Most recent lease purpose, used for diagnostics, never compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Role {
    Layer,
    Scratch,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Trimmed {
    pub total: usize,
    pub scratch: usize,
}

#[derive(Default)]
pub(crate) struct Pool {
    free: FreePool<Target>,
}

impl Pool {
    /// Returns a valid poolable backing extent for `requested_size`, if possible
    /// without exceeding the passed 2D texture size limit.
    pub fn backing_extent(requested_size: Size<u32>, maximum_dimension: u32) -> Option<Size<u32>> {
        Some(SizeClass::new(requested_size, maximum_dimension)?.extent())
    }

    pub fn lease_layer(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        requested_size: Size<u32>,
    ) -> (Target, bool) {
        let extent = Self::backing_extent(requested_size, device.limits().max_texture_dimension_2d)
            .expect("isolated layer target viewport must fit the device texture limit");

        self.lease(device, format, extent, Role::Layer)
    }

    pub fn lease_scratch(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        requested_size: Size<u32>,
    ) -> Result<(Target, bool), ScratchError> {
        let extent =
            scratch::backing_extent(requested_size, device.limits().max_texture_dimension_2d)?;
        Ok(self.lease(device, format, extent, Role::Scratch))
    }

    fn lease(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        extent: Size<u32>,
        role: Role,
    ) -> (Target, bool) {
        if let Some(mut target) = self.free.take(Key::new(format, extent)) {
            target.last_role = role;
            return (target, true);
        }

        (Target::new(device, format, extent, role), false)
    }

    pub fn release(&mut self, target: Target) {
        let key = Key::new(target.format, target.extent);
        let role = target.last_role;
        self.free.release(key, role, target);
    }

    pub fn bytes(&self) -> u64 {
        self.free.bytes
    }

    #[cfg(test)]
    pub fn scratch_bytes(&self) -> u64 {
        self.free.scratch_bytes
    }

    pub(super) fn trim_to_bytes(&mut self, maximum: u64, budget: u64) -> Trimmed {
        self.free.trim_to_bytes(maximum, budget)
    }

    #[cfg(test)]
    fn free_targets(&self) -> impl Iterator<Item = &Target> {
        self.free
            .buckets
            .values()
            .flat_map(|bucket| bucket.entries.iter().map(|entry| &entry.target))
    }
}

/// Exact compatibility for the current 2D, single-sample, single-mip targets.
/// Role is deliberately absent: scratch and layer captures share backings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl Key {
    fn new(format: wgpu::TextureFormat, extent: Size<u32>) -> Self {
        Self {
            width: extent.width,
            height: extent.height,
            format,
        }
    }

    fn bytes(self) -> u64 {
        texture_byte_size(self.format, Size::new(self.width, self.height))
    }
}

struct Free<T> {
    target: T,
    role: Role,
    released_at: u64,
}

struct Bucket<T> {
    bytes: u64,
    entries: VecDeque<Free<T>>,
}

/// The production reuse policy, generic only so CPU tests need no GPU handles.
struct FreePool<T> {
    buckets: FxHashMap<Key, Bucket<T>>,
    bytes: u64,
    scratch_bytes: u64,
    next_release: u64,
    // Dense bucket heads, populated only under pressure. Capacity is reused.
    candidates: Vec<(Key, u64)>,
}

impl<T> Default for FreePool<T> {
    fn default() -> Self {
        Self {
            buckets: FxHashMap::default(),
            bytes: 0,
            scratch_bytes: 0,
            next_release: 0,
            candidates: Vec::new(),
        }
    }
}

impl<T> FreePool<T> {
    fn take(&mut self, key: Key) -> Option<T> {
        let bucket = self.buckets.get_mut(&key)?;
        let entry = bucket.entries.pop_back()?;
        let bytes = bucket.bytes;
        self.remove_usage(bytes, entry.role);
        // Keep the empty deque for a return later in this frame.
        Some(entry.target)
    }

    fn release(&mut self, key: Key, role: Role, target: T) {
        if self.next_release == u64::MAX {
            // Extremely rare: preserve global order instead of wrapping stamps.
            let mut entries: Vec<_> = self
                .buckets
                .values_mut()
                .flat_map(|bucket| bucket.entries.iter_mut())
                .collect();
            entries.sort_unstable_by_key(|entry| entry.released_at);
            self.next_release = entries.len() as u64;
            for (index, entry) in entries.into_iter().enumerate() {
                entry.released_at = index as u64;
            }
        }

        let bucket = self.buckets.entry(key).or_insert_with(|| Bucket {
            bytes: key.bytes(),
            entries: VecDeque::new(),
        });
        bucket.entries.push_back(Free {
            target,
            role,
            released_at: self.next_release,
        });
        self.next_release += 1;
        self.bytes += bucket.bytes;
        if role == Role::Scratch {
            self.scratch_bytes += bucket.bytes;
        }
    }

    fn remove_usage(&mut self, bytes: u64, role: Role) {
        self.bytes -= bytes;
        if role == Role::Scratch {
            self.scratch_bytes -= bytes;
        }
    }

    /// Frame-end pressure trim. `maximum` is the free pool's allowance after
    /// retained ownership; `budget` is the entire renderer's configured budget.
    fn trim_to_bytes(&mut self, maximum: u64, budget: u64) -> Trimmed {
        self.trim_to_bytes_with(maximum, budget, drop)
    }

    fn trim_to_bytes_with(
        &mut self,
        maximum: u64,
        budget: u64,
        mut discard: impl FnMut(T),
    ) -> Trimmed {
        debug_assert!(
            maximum <= budget,
            "free allowance cannot exceed the shared budget"
        );
        let mut trimmed = Trimmed::default();
        if self.bytes > maximum {
            // These cannot fit even if all other ownership is discarded. Drop
            // them first so they cannot flush reusable smaller backings.
            self.buckets.retain(|_, bucket| {
                if bucket.bytes > budget {
                    for entry in bucket.entries.drain(..) {
                        self.bytes -= bucket.bytes;
                        let scratch = entry.role == Role::Scratch;
                        if scratch {
                            self.scratch_bytes -= bucket.bytes;
                        }
                        trimmed.total += 1;
                        trimmed.scratch += usize::from(scratch);
                        discard(entry.target);
                    }
                    false
                } else {
                    true
                }
            });

            if self.bytes > maximum {
                self.candidates
                    .extend(self.buckets.iter().filter_map(|(&key, bucket)| {
                        bucket.entries.front().map(|entry| (key, entry.released_at))
                    }));
                while self.bytes > maximum {
                    let index = self
                        .candidates
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, stamp))| *stamp)
                        .map(|(index, _)| index)
                        .expect("nonempty free pool has an eviction candidate");
                    let key = self.candidates[index].0;
                    let bucket = self.buckets.get_mut(&key).expect("indexed bucket");
                    let entry = bucket.entries.pop_front().expect("indexed oldest entry");
                    let bytes = bucket.bytes;
                    if let Some(next) = bucket.entries.front() {
                        self.candidates[index].1 = next.released_at;
                    } else {
                        let _ = self.candidates.swap_remove(index);
                    }
                    self.remove_usage(bytes, entry.role);
                    trimmed.total += 1;
                    trimmed.scratch += usize::from(entry.role == Role::Scratch);
                    // Drop ownership, never destroy a texture that encoded GPU
                    // commands may still reference.
                    discard(entry.target);
                }
                self.candidates.clear();
            }
        }

        // No idle TTL. Only empty metadata is reclaimed below the budget.
        self.buckets.retain(|_, bucket| !bucket.entries.is_empty());
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_is_pointer_sized_and_ownership_drops_once() {
        use std::cell::Cell;
        use std::rc::Rc;
        struct Dropped(Rc<Cell<usize>>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        assert_eq!(std::mem::size_of::<Target>(), std::mem::size_of::<usize>());
        let drops = Rc::new(Cell::new(0));
        let mut pool = FreePool::default();
        let key = Key::new(wgpu::TextureFormat::Rgba8Unorm, Size::new(1, 1));
        let lease = Lease(Box::new(Dropped(Rc::clone(&drops))));
        let address = std::ptr::from_ref::<Dropped>(&lease);
        pool.release(key, Role::Layer, lease);
        let lease = pool.take(key).unwrap();
        assert_eq!(std::ptr::from_ref::<Dropped>(&lease), address);
        assert!(pool.take(key).is_none());
        assert_eq!(pool.trim_to_bytes(0, 0), Trimmed::default());
        assert_eq!(drops.get(), 0);
        pool.release(key, Role::Scratch, lease);
        assert_eq!(
            pool.trim_to_bytes(0, 0),
            Trimmed {
                total: 1,
                scratch: 1
            }
        );
        assert_eq!(drops.get(), 1);
        pool.release(
            key,
            Role::Layer,
            Lease(Box::new(Dropped(Rc::clone(&drops)))),
        );
        let detached = pool.take(key).unwrap();
        drop(pool);
        assert_eq!(drops.get(), 1);
        drop(detached);
        assert_eq!(drops.get(), 2);
    }

    #[test]
    fn size_classes_are_normalized_in_increment_units() {
        assert_eq!(
            SizeClass::new(Size::new(1, 64), 4096),
            Some(SizeClass {
                width: 1,
                height: 1,
            })
        );
        assert_eq!(
            SizeClass::new(Size::new(65, 128), 4096),
            Some(SizeClass {
                width: 2,
                height: 2,
            })
        );
        assert_eq!(
            Pool::backing_extent(Size::new(1, 65), 4096),
            Some(Size::new(64, 128))
        );
    }

    #[test]
    fn terminal_size_class_does_not_cross_the_device_limit() {
        let maximum = 4096;
        let first = SizeClass::new(Size::new(maximum - SIZE_INCREMENT + 1, maximum), maximum)
            .expect("terminal size class");
        let last =
            SizeClass::new(Size::new(maximum, maximum), maximum).expect("maximum size class");

        assert_eq!(first, last);
        assert_eq!(last.extent(), Size::new(maximum, maximum));
        assert_eq!(
            Pool::backing_extent(Size::new(maximum + 1, 1), maximum),
            None
        );
        assert_eq!(Pool::backing_extent(Size::new(0, 1), maximum), None);
        // Layers require a complete size class even at a non-aligned limit.
        assert_eq!(Pool::backing_extent(Size::new(65, 1), 100), None);
    }

    #[test]
    fn texture_accounting_uses_the_actual_render_format() {
        let extent = Size::new(64, 32);

        assert_eq!(
            texture_byte_size(wgpu::TextureFormat::Rgba8Unorm, extent),
            64 * 32 * 4
        );
        assert_eq!(
            texture_byte_size(wgpu::TextureFormat::Rgba16Float, extent),
            64 * 32 * 8
        );
    }
}
