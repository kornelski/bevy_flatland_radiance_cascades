// This is a hacky way of dumping buffers into the window. There is nothing interesting here.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(2) @binding(0) var base_color_texture: texture_2d<f32>;
@group(2) @binding(1) var base_color_sampler: sampler;
@group(2) @binding(2) var<uniform> mode: u32;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4f {
    let data = textureSample(base_color_texture, base_color_sampler, mesh.uv);
    var val: vec2f;
    if mode == 0 {
        val = vec2(max(data.b, max(data.r, data.g)), data.a);
    } else {
        return vec4(pow((vec3f(0.001)+data.rgb) * data.a, vec3f(1./2.4)), 1.);
    }
    let a = abs(val);
    return vec4(max(a.r, a.g)*0.1, val, 1.);
}
