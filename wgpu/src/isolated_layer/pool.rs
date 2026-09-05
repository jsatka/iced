//! Renderer-owned texture targets and their reuse pool.

use crate::core::Size;
use crate::text;

/// Physical-pixel increment used to quantize each pooled backing-texture dimension.
const SIZE_INCREMENT: u32 = 64;

/// A normalized two-dimensional backing-texture size.
///
/// Dimensions are stored in units of [`SIZE_INCREMENT`] so pool lookups only
/// need to compare four bytes of size-class data.
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

/// A renderer-owned texture target.
pub(crate) struct Target {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub extent: Size<u32>,
    pub text_viewport: text::Viewport,
    size_class: SizeClass,
    format: wgpu::TextureFormat,
    last_used: u64,
}

impl Target {
    pub fn byte_size(&self) -> u64 {
        texture_byte_size(self.format, self.extent)
    }

    fn new(
        device: &wgpu::Device,
        text_pipeline: &text::Pipeline,
        format: wgpu::TextureFormat,
        size_class: SizeClass,
        frame: u64,
    ) -> Self {
        let extent = size_class.extent();
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
            text_viewport: text_pipeline.create_viewport(device),
            size_class,
            format,
            last_used: frame,
        }
    }
}

fn texture_byte_size(format: wgpu::TextureFormat, extent: Size<u32>) -> u64 {
    let (block_width, block_height) = format.block_dimensions();
    let bytes_per_block = format
        .block_copy_size(None)
        .expect("GPU color targets must have a defined texel-block size");

    u64::from(extent.width.div_ceil(block_width))
        .saturating_mul(u64::from(extent.height.div_ceil(block_height)))
        .saturating_mul(u64::from(bytes_per_block))
}

#[derive(Default)]
pub(crate) struct Pool {
    free: Vec<Target>,
}

impl Pool {
    /// Returns a valid poolable backing extent for `requested_size`, if possible
    /// without exceeding the passed 2D texture size limit.
    pub fn backing_extent(requested_size: Size<u32>, maximum_dimension: u32) -> Option<Size<u32>> {
        Some(SizeClass::new(requested_size, maximum_dimension)?.extent())
    }

    pub fn lease(
        &mut self,
        device: &wgpu::Device,
        text_pipeline: &text::Pipeline,
        format: wgpu::TextureFormat,
        requested_size: Size<u32>,
        frame: u64,
    ) -> (Target, bool) {
        let size_class = SizeClass::new(requested_size, device.limits().max_texture_dimension_2d)
            .expect("isolated-layer target viewport must fit the device texture limit");

        if let Some(index) = self
            .free
            .iter()
            .position(|target| target.size_class == size_class && target.format == format)
        {
            let mut target = self.free.swap_remove(index);
            target.last_used = frame;
            return (target, true);
        }

        (
            Target::new(device, text_pipeline, format, size_class, frame),
            false,
        )
    }

    pub fn release(&mut self, mut target: Target, frame: u64) {
        target.last_used = frame;
        self.free.push(target);
    }

    pub fn bytes(&self) -> u64 {
        self.free
            .iter()
            .fold(0, |bytes, target| bytes.saturating_add(target.byte_size()))
    }

    pub fn trim_idle(&mut self, frame: u64) -> usize {
        let before = self.free.len();
        self.free
            .retain(|target| frame.wrapping_sub(target.last_used) <= 120);
        before - self.free.len()
    }

    pub fn trim_to_bytes(&mut self, maximum: u64, frame: u64) -> usize {
        self.free
            .sort_unstable_by_key(|target| frame.wrapping_sub(target.last_used));
        let mut bytes = self.bytes();
        let mut removed = 0;

        while bytes > maximum && !self.free.is_empty() {
            let target = self.free.pop().expect("free pool checked non-empty");
            bytes = bytes.saturating_sub(target.byte_size());
            removed += 1;
        }

        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
