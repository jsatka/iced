mod alpha_mask;
mod drop_shadow;
mod dual_kawase_blur;
mod gaussian_blur;
mod pipeline;

pub use alpha_mask::AlphaMask;
pub use drop_shadow::DropShadow;
pub use dual_kawase_blur::DualKawaseBlur;
pub use gaussian_blur::GaussianBlur;

/// Blur algorithm and settings used by an isolated layer drop shadow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Blur {
    /// Separable Gaussian filtering.
    Gaussian(GaussianBlur),
    /// Downsample/upsample Dual Kawase filtering.
    DualKawase(DualKawaseBlur),
}

impl Default for Blur {
    fn default() -> Self {
        Self::Gaussian(GaussianBlur::default())
    }
}

impl From<GaussianBlur> for Blur {
    fn from(settings: GaussianBlur) -> Self {
        Self::Gaussian(settings)
    }
}

impl From<DualKawaseBlur> for Blur {
    fn from(settings: DualKawaseBlur) -> Self {
        Self::DualKawase(settings)
    }
}

fn canonical(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value.is_nan() {
        return minimum;
    }

    let value = value.clamp(minimum, maximum);
    if value == 0.0 { 0.0 } else { value }
}
