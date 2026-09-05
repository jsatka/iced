struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
struct Params { geometry: vec4<f32>, parameters: vec4<f32> }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

fn sample_source(uv: vec2<f32>) -> vec4<f32> {
    let valid_uv = params.geometry.xy;
    let half_texel = params.geometry.zw * 0.5;
    return textureSample(source, source_sampler, clamp(uv * valid_uv, half_texel, valid_uv - half_texel));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sigma = max(params.parameters.z, 0.001);
    let radius = min(i32(ceil(sigma * 3.0)), 24);
    let stride = max(1.0, sigma * 3.0 / 24.0);
    let direction = params.parameters.xy * params.geometry.zw;
    var result = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = -24; i <= 24; i = i + 1) {
        if (abs(i) <= radius) {
            let x = f32(i) * stride;
            let weight = exp(-(x * x) / (2.0 * sigma * sigma));
            result += sample_source(input.uv + direction * x / params.geometry.xy) * weight;
            total += weight;
        }
    }
    return result / max(total, 0.0001);
}
