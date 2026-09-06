struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
struct Params {
    geometry: vec4<f32>,
    parameters: vec4<f32>,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var kernel: texture_2d<f32>;

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
    let step = params.parameters.xy * params.geometry.zw / params.geometry.xy;
    var result = sample_source(input.uv) * params.parameters.z;
    for (var i = 0u; i < u32(params.parameters.w); i += 1u) {
        // Each entry combines two adjacent texels using linear interpolation.
        let pair = textureLoad(kernel, vec2<i32>(i32(i), 0), 0).xy;
        let offset = step * pair.x;
        result += (sample_source(input.uv + offset) + sample_source(input.uv - offset)) * pair.y;
    }
    return result;
}
