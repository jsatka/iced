mod alpha_mask;
mod drop_shadow;
mod gaussian_blur;
mod pipeline;

pub use alpha_mask::AlphaMask;
pub use drop_shadow::DropShadow;
pub use gaussian_blur::GaussianBlur;

fn canonical(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value.is_nan() {
        return minimum;
    }

    let value = value.clamp(minimum, maximum);
    if value == 0.0 { 0.0 } else { value }
}
