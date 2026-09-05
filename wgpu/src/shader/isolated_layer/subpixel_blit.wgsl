struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

struct Params {
    // Source and destination origins in their respective physical-pixel spaces.
    origins: vec4<f32>,
    // Valid source extent followed by the full backing extent.
    source_geometry: vec4<f32>,
    opacity: vec4<f32>,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let source_position = input.position.xy - params.origins.zw + params.origins.xy;
    let valid_extent = params.source_geometry.xy;
    let backing_extent = params.source_geometry.zw;
    let clamped_position = clamp(
        source_position,
        vec2<f32>(0.5),
        valid_extent - vec2<f32>(0.5),
    );
    let edge_weight = saturate(
        min(source_position, valid_extent - source_position) + vec2<f32>(0.5),
    );

    return textureSample(source, source_sampler, clamped_position / backing_extent)
        * edge_weight.x
        * edge_weight.y
        * params.opacity.x;
}
