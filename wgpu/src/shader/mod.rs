//! Built-in renderer shader sources.

/// Shader sources for built-in isolated layer effects.
pub mod isolated_layer {
    /// Shader sources for programmable isolated layer effects.
    pub mod effect {
        /// Gaussian blur shader source.
        pub const GAUSSIAN: &str = include_str!("isolated_layer/effect/gaussian.wgsl");

        /// Dual Kawase downsample, upsample, identity blend, and fused shadow source.
        pub const DUAL_KAWASE: &str = include_str!("isolated_layer/effect/dual_kawase.wgsl");

        /// Alpha mask shader source.
        pub const MASK: &str = include_str!("isolated_layer/effect/mask.wgsl");

        /// Drop shadow shader source.
        pub const SHADOW: &str = include_str!("isolated_layer/effect/shadow.wgsl");
    }
}
