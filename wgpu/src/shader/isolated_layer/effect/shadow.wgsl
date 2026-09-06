struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>}
struct Params {
    geometry: vec4<f32>,
    parameters: vec4<f32>,
    color: vec4<f32>}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var blurred: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

fn sample_image(image: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let valid_uv = params.geometry.xy;
    let half_texel = params.geometry.zw * 0.5;
    return textureSample(image, source_sampler, clamp(uv * valid_uv, half_texel, valid_uv - half_texel));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let original = sample_image(source, input.uv);
    let offset_uv = params.parameters.xy * params.geometry.zw / params.geometry.xy;
    let alpha = sample_image(blurred, input.uv - offset_uv).a;
    let shadow_alpha = alpha * params.color.a;
    let shadow = vec4<f32>(params.color.rgb * shadow_alpha, shadow_alpha);
    return original + shadow * (1.0 - original.a);
}
