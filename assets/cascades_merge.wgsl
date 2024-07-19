#import "cascades_common.wgsl"::{CascadesGlobals, CascadesParams, PROBE_STORAGE_COLUMN_WIDTH};

@group(0) @binding(0) var<uniform> globals: CascadesGlobals;
@group(0) @binding(1) var<uniform> params: CascadesParams;

@group(1) @binding(0) var cascade_merge_output: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var cascades: texture_storage_2d<rgba32float, read>;
@group(1) @binding(2) var cascade_merge_input: texture_storage_2d<rgba32float, read>;

@compute @workgroup_size(8, 8, 1)
fn cascades_merge_m0(@builtin(global_invocation_id) invocation_id: vec3u) {
    let output_location = invocation_id.xy;

    let stride = vec2(
        PROBE_STORAGE_COLUMN_WIDTH,
        globals.initial_angles / PROBE_STORAGE_COLUMN_WIDTH
    );

    // stretches ignoring aspect ratio, so initial_spacing is irrelevant
    // 1.1 to leave 0.5 of probe space at each edge
    let num_probes = textureDimensions(cascade_merge_input) / stride + vec2(1);

    // TODO: pick the right neigbhors
    let probe_index_f = vec2f(output_location * num_probes) / vec2f(textureDimensions(cascade_merge_output));
    // -0.5 and rounding down picks top-left neighbor
    let probe_index = vec2u(probe_index_f - vec2f(0.5));

    let p0 = probe_index * stride;
    var p1 = p0 + vec2(stride.x, 0);
    var p2 = p0 + vec2(0, stride.y);
    var p3 = p0 + stride;

    var rays = vec4f(0.);
    for(var angle = 0u; angle < globals.initial_angles; angle++) {
        let angle_pos = vec2u(angle & (PROBE_STORAGE_COLUMN_WIDTH - 1), angle / PROBE_STORAGE_COLUMN_WIDTH);

        rays += textureLoad(cascade_merge_input, p0 + angle_pos)
              + textureLoad(cascade_merge_input, p1 + angle_pos)
              + textureLoad(cascade_merge_input, p2 + angle_pos)
              + textureLoad(cascade_merge_input, p3 + angle_pos);
    }

    textureStore(cascade_merge_output, output_location, rays / f32(globals.initial_angles * 4));
}

struct UnmergedCascadeProbe {
    storage_location: vec2u,
    stride: vec2u,
    num_angles: u32,
}

fn cascade_probe_spacing(cascade: u32) -> UnmergedCascadeProbe {
    let num_angles = globals.initial_angles << (cascade * globals.branching_factor);
    var stride: vec2u;
    if globals.branching_factor == 1 {
        stride = vec2(
            PROBE_STORAGE_COLUMN_WIDTH,
            num_angles / PROBE_STORAGE_COLUMN_WIDTH, // TODO: inaccurate for larger branching factors
        );
    } else {
        let num_angles_sqrt = globals.initial_angles << (cascade * (globals.branching_factor/2));
        stride = vec2(num_angles_sqrt);
    }

    var cascade_storage_offset_x = 0u;
    if cascade > 0 {
        if globals.branching_factor == 1 {
            let storage_width = globals.world_size.x / globals.initial_spacing * PROBE_STORAGE_COLUMN_WIDTH;
            cascade_storage_offset_x = storage_width - (storage_width >> (cascade - 1u));
        } else {
            cascade_storage_offset_x = params.cascade * (globals.world_size.x / globals.initial_spacing * stride.x);
        }
    }

    return UnmergedCascadeProbe(vec2u(cascade_storage_offset_x, 0), stride, num_angles);
}

fn angle_pos(angle_index: u32) -> vec2u {
    return vec2(
        // TODO: invalid for higher branching factor
        angle_index & (PROBE_STORAGE_COLUMN_WIDTH - 1),
        angle_index / PROBE_STORAGE_COLUMN_WIDTH,
    );
}

@compute @workgroup_size(8, 8, 1)
fn cascades_merge_m1(@builtin(global_invocation_id) invocation_id: vec3u) {
    let probe_index = invocation_id.xy;
    let this_cascade = cascade_probe_spacing(params.cascade);
    let next_cascade = cascade_probe_spacing(params.cascade + 1);

    let this_probe = this_cascade.storage_location
        + this_cascade.stride * probe_index;

    // -1 and rounding picks top-left corner that may be above or left
    let next_probe_index = (probe_index - vec2(1)) / 2;
    let next_probe_0 = next_cascade.storage_location + next_cascade.stride * next_probe_index;
    let next_probe_1 = next_probe_0 + vec2(next_cascade.stride.x, 0);
    let next_probe_2 = next_probe_0 + vec2(0, next_cascade.stride.y);
    let next_probe_3 = next_probe_0 + next_cascade.stride;

    // cascades are never perfectly aligned
    let blend = vec2f(
        select(0.6666, 0.3333, (probe_index.x & 1) == 0),
        select(0.6666, 0.3333, (probe_index.y & 1) == 0),
    );
    // This is probably stupid. Shouldn't it be some weighed euclidean distance?
    var blend_0 = blend.x * blend.y;
    var blend_1 = (1. - blend.x) * blend.y;
    var blend_2 = blend.x * (1. - blend.y);
    var blend_3 = (1. - blend.x) * (1. - blend.y);

    blend_0 = blend_0 * blend_0;
    blend_1 = blend_1 * blend_1;
    blend_2 = blend_2 * blend_2;
    blend_3 = blend_3 * blend_3;

    let ratio = next_cascade.num_angles / this_cascade.num_angles;
    for(var i=0u; i < this_cascade.num_angles; i++) {
        let this_probe_angle = this_probe + angle_pos(i);

        var from_next = vec3f(0.);
        if params.cascade < globals.num_cascades - 1 {
            for(var j=0u; j < ratio; j++) {
                let jidx = (i*ratio + j + next_cascade.num_angles ) % next_cascade.num_angles;
                // read the same ray angle from 4 spatial neighbors
                // TODO: are the angles offset ok? should they be interpolated?
                let next_angle_pos = angle_pos(jidx);
                // it'd be nice to rearrange the cascades (or just merge?) to support hw interpolation
                from_next += (
                    blend_0 * textureLoad(cascade_merge_input, next_probe_0 + next_angle_pos).rgb +
                    blend_1 * textureLoad(cascade_merge_input, next_probe_1 + next_angle_pos).rgb +
                    blend_2 * textureLoad(cascade_merge_input, next_probe_2 + next_angle_pos).rgb +
                    blend_3 * textureLoad(cascade_merge_input, next_probe_3 + next_angle_pos).rgb
                );
            }
        }
        var combined = textureLoad(cascades, this_probe_angle);
        combined += combined.a * vec4(from_next, 0.);

        textureStore(cascade_merge_output, this_probe_angle, combined);
    }
}
