struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
struct Params {
    source: vec4<f32>,
    original: vec4<f32>,
    sampling: vec4<f32>,
    output: vec4<f32>,
    color: vec4<f32>,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var original: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var result: VertexOutput;
    result.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    result.uv = uv;
    return result;
}

// Source-texel coordinates, centers at n + 0.5. Clamping to valid centers
// avoids pooled padding; coverage restores transparent bilinear edge extension.
fn sample_image(image: texture_2d<f32>, position: vec2<f32>, geometry: vec4<f32>) -> vec4<f32> {
    let coverage = clamp(position + 0.5, vec2<f32>(0.0), vec2<f32>(1.0))
        * clamp(geometry.xy + 0.5 - position, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv = clamp(position, vec2<f32>(0.5), geometry.xy - 0.5) / geometry.zw;
    return textureSampleLevel(image, image_sampler, uv, 0.0) * coverage.x * coverage.y;
}

fn read_source(position: vec2<f32>) -> vec4<f32> {
    return sample_image(source, position, params.source);
}

fn downsample(position: vec2<f32>) -> vec4<f32> {
    let h = params.sampling.xy * 0.5;
    return (read_source(position) * 4.0
        + read_source(position + h)
        + read_source(position - h)
        + read_source(position + vec2<f32>(h.x, -h.y))
        + read_source(position + vec2<f32>(-h.x, h.y))) / 8.0;
}

fn upsample(position: vec2<f32>) -> vec4<f32> {
    let h = params.sampling.xy * 0.5;
    return (read_source(position + vec2<f32>(2.0 * h.x, 0.0))
        + read_source(position - vec2<f32>(2.0 * h.x, 0.0))
        + read_source(position + vec2<f32>(0.0, 2.0 * h.y))
        + read_source(position - vec2<f32>(0.0, 2.0 * h.y))
        + 2.0 * (read_source(position + h)
            + read_source(position - h)
            + read_source(position + vec2<f32>(h.x, -h.y))
            + read_source(position + vec2<f32>(-h.x, h.y)))) / 12.0;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let mode = params.sampling.w;
    if mode < 0.5 {
        return downsample(input.uv * params.source.xy);
    }
    if mode < 1.5 {
        if params.sampling.z >= 1.0 { return upsample(input.uv * params.source.xy); }
        let base = sample_image(original, input.uv * params.original.xy, params.original);
        if params.sampling.z == 0.0 { return base; }
        return mix(base, upsample(input.uv * params.source.xy), params.sampling.z);
    }
    let base = sample_image(original, input.uv * params.original.xy, params.original);
    if mode < 2.5 {
        if params.sampling.z == 0.0 { return base; }
        return mix(base, upsample(input.uv * params.source.xy), params.sampling.z);
    }
    let shifted_uv = input.uv - params.output.zw / params.output.xy;
    var alpha = sample_image(original, shifted_uv * params.original.xy, params.original).a;
    if params.sampling.z > 0.0 {
        alpha = mix(alpha, upsample(shifted_uv * params.source.xy).a, params.sampling.z);
    }
    let shadow_alpha = alpha * params.color.a;
    let shadow = vec4<f32>(params.color.rgb * shadow_alpha, shadow_alpha);
    return base + shadow * (1.0 - base.a);
}
