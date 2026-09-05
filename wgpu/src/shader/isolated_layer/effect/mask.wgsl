struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>}
struct Params {
    geometry: vec4<f32>,
    viewport: vec4<f32>,
    edges: vec4<f32>}
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
    let half_texel = params.geometry.zw * 0.5;
    return textureSample(source, source_sampler, clamp(uv * params.geometry.xy, half_texel, params.geometry.xy - half_texel));
}

fn edge_fade(distance: f32, extent: f32) -> f32 {
    if extent <= 0.0 { return 1.0; }
    return smoothstep(0.0, extent, distance);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = input.uv * params.viewport.xy;
    let alpha = edge_fade(pixel.y, params.edges.x)
        * edge_fade(params.viewport.x - pixel.x, params.edges.y)
        * edge_fade(params.viewport.y - pixel.y, params.edges.z)
        * edge_fade(pixel.x, params.edges.w);
    return sample_source(input.uv) * alpha;
}
