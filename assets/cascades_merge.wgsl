#import "cascades_common.wgsl"::{CascadesGlobals, CascadesParams, BRANCHING_FACTOR, PROBE_STORAGE_COLUMN_WIDTH};

@group(0) @binding(0) var<uniform> globals: CascadesGlobals;
@group(0) @binding(1) var<uniform> params: CascadesParams;

@group(1) @binding(0) var cascade_merge_output: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var cascades: texture_storage_2d<rgba32float, read>;
@group(1) @binding(2) var cascade_merge_input: texture_storage_2d<rgba32float, read>;

/// merging arranges angle indexes in a square.
/// not all num_angles square cleanly, and the len is not always a power of 2
fn merge_num_angles_side_len(cascade: u32) -> u32 {
    let num_angles = globals.initial_angles << (cascade * BRANCHING_FACTOR);
    return u32(ceil(sqrt(f32(num_angles))));
}

struct UnmergedCascadeProbe {
    stride: vec2u,
    num_angles: u32,
}

fn cascade_probe_spacing(cascade: u32) -> UnmergedCascadeProbe {
    let num_angles = globals.initial_angles << (cascade * BRANCHING_FACTOR);
    var stride: vec2u;
    if BRANCHING_FACTOR == 1 {
        stride = vec2(
            PROBE_STORAGE_COLUMN_WIDTH,
            num_angles / PROBE_STORAGE_COLUMN_WIDTH, // TODO: inaccurate for larger branching factors
        );
    } else {
        let num_angles_sqrt = globals.initial_angles << (cascade * (BRANCHING_FACTOR/2));
        stride = vec2(num_angles_sqrt);
    }

    return UnmergedCascadeProbe(stride, num_angles);
}

fn cascade_angle_location_offset(angle_index: u32) -> vec2u {
    return vec2(
        // TODO: invalid for higher branching factor
        angle_index & (PROBE_STORAGE_COLUMN_WIDTH - 1),
        angle_index / PROBE_STORAGE_COLUMN_WIDTH,
    );
}

struct MergedProbe {
    probe_index: vec2u,
    merged_angle_index: u32,
    unmerged_angle_ratio: u32,
}

fn direction_first_merged_angles_stride(cascade: u32) -> vec2u {
    return ((globals.world_size / globals.initial_spacing) >> vec2(cascade));
     // + (3 + cascade * 3); // hacky hack
}

fn direction_first_merged_angle(cascade: u32, invocation_id: vec2u) -> MergedProbe {
    let angles_stride = direction_first_merged_angles_stride(cascade);
    // angles are space-adjacent for every probe
    let probe_index = invocation_id % angles_stride;

    let max_index = (globals.world_size / globals.initial_spacing) >> vec2u(cascade);
    if any(probe_index >= max_index) {
        return MergedProbe(vec2u(-1i), u32(-1i), 0);
    }

    let num_angles_sqrt = merge_num_angles_side_len(cascade);
    // angles are stored in a square, so the underlying buffer keeps screen's aspect ratio
    let merged_angle_xy = invocation_id / angles_stride;
    if any(merged_angle_xy >= vec2(num_angles_sqrt)) {
        return MergedProbe(vec2u(-1i), u32(-1i), 0);
    }
    let merged_angle_index = merged_angle_xy.x + merged_angle_xy.y * num_angles_sqrt;
    let unmerged_angle_ratio = 1u << BRANCHING_FACTOR;

    return MergedProbe(probe_index, merged_angle_index, unmerged_angle_ratio);
}

fn merged_angle_index_offset(cascade: u32, angle_index: u32) -> vec2u {
    let angles_stride = direction_first_merged_angles_stride(cascade);

    let num_angles = globals.initial_angles << (cascade * BRANCHING_FACTOR);
    // shouldn't happen
    if angle_index >= num_angles {
        return vec2u(-1i);
    }

    let num_angles_sqrt = merge_num_angles_side_len(cascade);
    return angles_stride * vec2u(
        angle_index % num_angles_sqrt, // can't &-1, not a power of 2
        angle_index / num_angles_sqrt,
    );
}

// dispatched per merged angle (merges multiple rays)
// this is for cascade mN where N is the last one, and it runs first,
// since cascades are merged from N first to 0 last.
@compute @workgroup_size(8, 8, 1)
fn cascades_merge_first(@builtin(global_invocation_id) invocation_id: vec3u) {
    let output_location = invocation_id.xy;
    let this_merge = direction_first_merged_angle(params.cascade, output_location);

    // shouldn't happen? in debug that's junk between data points
    if this_merge.unmerged_angle_ratio == 0 {
        textureStore(cascade_merge_output, output_location, vec4f(0.));
        return;
    }

    let max_index = (globals.world_size / globals.initial_spacing) >> vec2(params.cascade);
    // shouldn't happen?
    if any(vec2u(this_merge.probe_index) >= max_index) {
        textureStore(cascade_merge_output, output_location, vec4f(0.));
        return;
    }

    let this_cascade = cascade_probe_spacing(params.cascade);
    let cascade_probe_start_location = this_cascade.stride * this_merge.probe_index;

    let next_num_angles = globals.initial_angles << ((params.cascade+1) * BRANCHING_FACTOR);

    // off by 1?
    if this_merge.merged_angle_index * this_merge.unmerged_angle_ratio >= next_num_angles {
        textureStore(cascade_merge_output, output_location, vec4f(0.));
        return;
    }

    // add skybox or offscreen lights here?

    var combined_averaged = vec4f(0.);
    for (var j=0u; j < this_merge.unmerged_angle_ratio; j += 1u) {
        let cascade_angle_index = j + this_merge.merged_angle_index * this_merge.unmerged_angle_ratio;

        let cascade_angle_location = cascade_probe_start_location + cascade_angle_location_offset(cascade_angle_index);
        let traced = textureLoad(cascades, cascade_angle_location);
        combined_averaged += traced; // .a is useless here?
    }

    // combined_averaged is divided by unmerged_angle_ratio to it equal to one "ray",
    // and because the tracing of rays divides light by the step length,
    // the value stored here represents average of all the light inside the cone spanning the merged rays
    // (I'm not sure if this has off-by-one error from counting fences vs fence posts, due to not counting the "gap" between pre-averaged cones).
    textureStore(cascade_merge_output, output_location, combined_averaged * (1. / f32(this_merge.unmerged_angle_ratio)));
}

// dispatched per merged angle (merges multiple rays)
@compute @workgroup_size(8, 8, 1)
fn cascades_merge_m1(@builtin(global_invocation_id) invocation_id: vec3u) {
    let output_location = invocation_id.xy;
    let this_merge = direction_first_merged_angle(params.cascade, output_location);

    // shouldn't happen?
    if this_merge.unmerged_angle_ratio == 0 {
        textureStore(cascade_merge_output, output_location, vec4f(0.));
        return;
    }

    // find a neighbor in cascade+1
    // -1 and rounding picks top-left corner that may be above or left
    let next_probe_index = (vec2i(this_merge.probe_index) - 1i) / 2;
    let next_num_probes = vec2i((globals.world_size / globals.initial_spacing) >> vec2(params.cascade + 1u));
    if any(next_probe_index+vec2(1i) >= next_num_probes) {
        return;
    }

    // cascades are never perfectly aligned
    let blend = vec2f(
        select(0.6666, 0.3333, (this_merge.probe_index.x & 1) == 0),
        select(0.6666, 0.3333, (this_merge.probe_index.y & 1) == 0),
    );
    // these add up to 1
    var blend_0 = blend.x * blend.y;
    var blend_1 = (1. - blend.x) * blend.y;
    var blend_2 = blend.x * (1. - blend.y);
    var blend_3 = (1. - blend.x) * (1. - blend.y);

    let this_cascade = cascade_probe_spacing(params.cascade);
    let next_num_angles = globals.initial_angles << ((params.cascade+1) * BRANCHING_FACTOR);
    let cascade_probe_start_location = this_cascade.stride * this_merge.probe_index;

    var combined_averaged = vec4f(0.);
    for (var j=0u; j < this_merge.unmerged_angle_ratio; j += 1u) {
        let cascade_angle_index = j + this_merge.merged_angle_index * this_merge.unmerged_angle_ratio;

        if cascade_angle_index >= next_num_angles {
            textureStore(cascade_merge_output, output_location, vec4f(0.));
            return;
        }

        // it's +, because direction-first
        let next_probe_location = next_probe_index + vec2i(merged_angle_index_offset(params.cascade+1, cascade_angle_index));


        // from_next represents a fraction of next cascade's num rays,
        // and traced represents a fraction of this cascade's num rays,
        // so they have different amount of "rays" in them, and therefore
        // a disproportional amounts of light.
        // the ratio between cascades is costant, so from_next's light intensity is scaled
        // by dividing by unmerged_angle_ratio AGAIN (for the 3rd time!).
        // - it was first divided when pre-averaging its cascade to have same scale as one ray of that next cascade
        // - then the second time to rescale it from ray of cascade with more rays to ray of cascade with fewer rays
        // - and then will be divided for the third time when this cascade gets pre-averaged. It sounds dodgy, but I think it's right.
        let from_next = (
            // TODO: bounds check?
            // direction-first storage means they definitely are next to each other
            blend_0 * textureLoad(cascade_merge_input, next_probe_location).rgb +
            blend_1 * textureLoad(cascade_merge_input, next_probe_location + vec2i(1, 0)).rgb +
            blend_2 * textureLoad(cascade_merge_input, next_probe_location + vec2i(0, 1)).rgb +
            blend_3 * textureLoad(cascade_merge_input, next_probe_location + vec2i(1, 1)).rgb
        ) *  (1. / f32(this_merge.unmerged_angle_ratio));


        let cascade_angle_location = cascade_probe_start_location + cascade_angle_location_offset(cascade_angle_index);
        let traced = textureLoad(cascades, cascade_angle_location);
        combined_averaged += traced + vec4f(traced.a * from_next, 0.);
    }
    // this is divided by unmerged_angle_ratio to normalize merged itensity to equivalent of 1 ray
    // TODO: not sure if it should be divided by 2 because it adds traced, or not?
    textureStore(cascade_merge_output, output_location, combined_averaged * (1. / f32(this_merge.unmerged_angle_ratio)));
}

// converts the last few angles into displayable lit(ish) pixels
@compute @workgroup_size(8, 8, 1)
fn cascades_merge_m0(@builtin(global_invocation_id) invocation_id: vec3u) {
    let output_location = invocation_id.xy;

    let angles_stride = direction_first_merged_angles_stride(0u);
    let num_angles_sqrt = merge_num_angles_side_len(0u);

    let num_probes_0 = globals.world_size / globals.initial_spacing;
    let num_angles = globals.initial_angles << (params.cascade * BRANCHING_FACTOR);

    let output_resolution = textureDimensions(cascade_merge_output);
    let probe_index = vec2u(vec2f(output_location) / vec2f(output_resolution) * vec2f(num_probes_0));

    var combined = vec4f(0.);
    for(var angle_index = 0u; angle_index < globals.initial_angles; angle_index += 1u) {
        let angle_offset = angles_stride * vec2u(
            angle_index % num_angles_sqrt,
            angle_index / num_angles_sqrt,
        );
        // TODO: sample nicely
        combined += textureLoad(cascade_merge_input, probe_index + angle_offset);
    }
    combined *= 1. / f32(globals.initial_angles);

    textureStore(cascade_merge_output, output_location, vec4f(combined.rgb * combined.a, combined.a));
}
