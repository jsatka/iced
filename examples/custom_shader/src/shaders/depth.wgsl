var<private> positions: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, -1.0)
);

@group(0) @binding(0) var depth_texture: texture_depth_2d;

struct Uniforms {
    projection: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_color: vec4<f32>,
}

@group(0) @binding(1) var<uniform> uniforms: Uniforms;

struct Output {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) v_index: u32) -> Output {
    var out: Output;

    out.position = vec4<f32>(positions[v_index], 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(input: Output) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);

    if depth > .9999 {
        discard;
    }

    let c = 1.0 - depth;

    return vec4<f32>(c, c, c, 1.0);
}
