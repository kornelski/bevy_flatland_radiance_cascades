struct CascadesGlobals {
    num_cascades: u32,
    initial_angles: u32,
    initial_spacing: u32,
    branching_factor: u32,
    world_size: vec2u,
    time: f32,
    delta_time: f32,
    mouse_button: u32,
    mouse_position: vec2f,
}

struct CascadesParams {
    cascade: u32,
    steps: u32,
}

@group(0) @binding(0) var<uniform> globals: CascadesGlobals;
@group(0) @binding(1) var<uniform> params: CascadesParams;

@group(1) @binding(0) var cascade_merge_output: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var cascade_merge_input: texture_storage_2d<rgba32float, read>;

/// 1<<BRANCHING_FACTOR times more angles per cascade
/// 1 gives doubling, 2 gives quadrupling.
const BRANCHING_FACTOR: u32 = #{BRANCHING_FACTOR};

/// merging arranges angle indexes in a square.
/// not all num_angles square cleanly, and the len is not always a power of 2
fn merge_num_angles_side_len(cascade: u32) -> u32 {
    let num_angles = globals.initial_angles << (cascade * BRANCHING_FACTOR);
    return u32(ceil(sqrt(f32(num_angles))));
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
// this is for cascade cN where N is the last one, and it runs first,
// since cascades are merged from N first to 0 last.
@compute @workgroup_size(8, 8, 1)
fn cascades_cmax(@builtin(global_invocation_id) invocation_id: vec3u) {
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

    let next_num_angles = globals.initial_angles << ((params.cascade+1) * BRANCHING_FACTOR);

    // off by 1?
    if this_merge.merged_angle_index * this_merge.unmerged_angle_ratio >= next_num_angles {
        textureStore(cascade_merge_output, output_location, vec4f(0.));
        return;
    }

    // add skybox or offscreen lights here?

    // the spatial resolution keeps halving
    let world_spacing = f32(globals.initial_spacing << params.cascade);
    let world_location = vec2f(this_merge.probe_index) * world_spacing
        + vec2f(world_spacing * 0.25); // center the probe within its square

    var combined_averaged = vec4f(0.);
    for (var j=0u; j < this_merge.unmerged_angle_ratio; j += 1u) {
        let cascade_angle_index = j + this_merge.merged_angle_index * this_merge.unmerged_angle_ratio;

        let probe = Probe(world_location, cascade_angle_index, index_to_angle(cascade_angle_index, next_num_angles));
        let traced = cascades_trace(probe);
        combined_averaged += traced; // .a is useless here?
    }

    // combined_averaged is divided by unmerged_angle_ratio to it equal to one "ray",
    // and because the tracing of rays divides light by the step length,
    // the value stored here represents average of all the light inside the cone spanning the merged rays
    // (I'm not sure if this has off-by-one error from counting fences vs fence posts, due to not counting the "gap" between pre-averaged cones).
    textureStore(cascade_merge_output, output_location, combined_averaged * (1. / f32(this_merge.unmerged_angle_ratio)));
}

// dispatched per merged angle (merges multiple rays)
// traces and merges for cascades max-1 to 1
@compute @workgroup_size(8, 8, 1)
fn cascades_c1(@builtin(global_invocation_id) invocation_id: vec3u) {
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

    let next_num_angles = globals.initial_angles << ((params.cascade+1) * BRANCHING_FACTOR);

    // the spatial resolution keeps halving
    let world_spacing = f32(globals.initial_spacing << params.cascade);
    let world_location = vec2f(this_merge.probe_index) * world_spacing
        + vec2f(world_spacing * 0.25); // center the probe within its square

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


        let probe = Probe(world_location, cascade_angle_index, index_to_angle(cascade_angle_index, next_num_angles));
        let traced = cascades_trace(probe);
        combined_averaged += traced + vec4f(traced.a * from_next, 0.);
    }
    // this is divided by unmerged_angle_ratio to normalize merged itensity to equivalent of 1 ray
    // TODO: not sure if it should be divided by 2 because it adds traced, or not?
    textureStore(cascade_merge_output, output_location, combined_averaged * (1. / f32(this_merge.unmerged_angle_ratio)));
}

// converts the last few angles into displayable lit(ish) pixels
@compute @workgroup_size(8, 8, 1)
fn cascades_c0(@builtin(global_invocation_id) invocation_id: vec3u) {
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

///////////////////
///
struct Probe {
    world_location: vec2f,
    angle_index: u32,
    // theta, in radians
    angle: f32,
}

const TAU: f32 = 6.283185307179;

fn index_to_angle(angle_index: u32, num_angles: u32) -> f32 {
    // + 0.5 starts cascade 0 with an X instead of +
    return (f32(angle_index) + 0.5) * (TAU / f32(num_angles));
}

// // This is the previous frame
// fn sample_fluence(world_location: vec2f) -> vec3f {
//     let texture_scale = 1. / vec2f(globals.world_size) * vec2f(textureDimensions(previous_frame));
//     let tex_location = world_location * texture_scale;
//     return textureLoad(previous_frame, vec2u(tex_location)).rgb;
// }

const scattering_coeff: f32 = 0.5; // how much light bounces
const absorption_coeff: f32 = 0.5; // how much light is converted to heat
const extinction_coeff: f32 = scattering_coeff + absorption_coeff;

// trace from a single location towards a single angle
fn trace_radiance(probe: Probe, interval_start: f32, interval_len: f32) -> vec4f {
    let direction = vec2f(cos(probe.angle), -sin(probe.angle));
    let p0 = probe.world_location + direction * interval_start;
    let delta = direction * interval_len;
    let step = 1. / min(f32(params.steps)+0.5, interval_len);
    let step_length = step * interval_len; // dx of the integral

    var total_transmittance = 1.; // β
    var total_radiance = vec3f(0.); // L_(start, end)

    let steps_to_edge = select((vec2f(globals.world_size) - p0) / delta, -p0 / delta, vec2(delta.x < 0., delta.y < 0.));
    let end = min(1., min(steps_to_edge.x, steps_to_edge.y));

    // the i loop goes from 0 to 1 with fractional steps (most float precision is under 1)
    // but starts at half step to avoid ambiguity around i==1 and doubled sampling at cascade ends
    for(var i = step * 0.5; i <= end; i += step) {
        // p is the next point to sample
        let p = p0 + i * delta;

        // Amount of light added. Multiplied by step_length, because we assume
        // the lights have an area, and samples in between would see the same light source.
        // TODO: this should use a mipmap at a resolution appropriate for the step size and spacing
        // and also the mipmap should blur light sources to enforce minimum light size.
        var radiance = sample_light(p, probe.angle) * step_length;

        // Average density here. It needs to exp() to represent attenuation
        // over the step length.
        // TODO: this could have materials or report absorption and scattering separately
        // TODO: this should use a mipmap at resolution appropriate for the step size and spacing
        let density = sample_density(p);
        if density > 0. {
            // The absorption vs scattering is probably buggy

            // Beer-Lambert law
            let transmittance = exp(-extinction_coeff * density * step_length);

            // Not sure if the phase function (Henyey-Greenstein) makes any sense in 2D?
            // Previous frame would probably need to keep track of ray directions for that?

            // This is the total amount of light at this point, from the previous frame.
            // var fluence = sample_fluence(p);

            var scattering = min(density * scattering_coeff, 0.99);

            // TODO: cascade 0 stoarge format is no longer compatible since I've added pre-merge
            // this means to make walls bounce the light
            // if density > 0.25 {
            //     // try to sample the light before a wall
            //     fluence = max(fluence, max(sample_fluence(p - direction), sample_fluence(p - step * 0.5)));
            // }

            // not sure if this should be affected by transmittance, or is that
            // counting transmittance twice, since the previous frame already handled attentuation?
            // radiance += fluence * step_length * scattering;

            total_transmittance *= transmittance;

            // this would maybe help scenes with more walls
            // TODO: also could check the previous cascade, and avoid launching more rays
            // if total_transmittance < 0.01 && fract(globals.time) < 0.5 {
            //     break;
            // }
        }
        total_radiance += radiance * total_transmittance;
    }

    // last ray of last cascade should trace into "infinity" (cubemap of the horizon)
    return vec4f(total_radiance, total_transmittance);
}

// Each angle is dispatched separately!
fn cascades_trace(probe: Probe) -> vec4f {
    // This needs more careful calculation to guarantee angular resolution
    let start = BRANCHING_FACTOR * ((globals.initial_angles << params.cascade) - globals.initial_angles);
    let end = BRANCHING_FACTOR * ((globals.initial_angles << (params.cascade + 1)) - globals.initial_angles);

    let start_longer = ((probe.angle_index / 2) & 1) != 0;
    let end_longer = (probe.angle_index & 1) != 0;
    let ray_scale = 1.; //4. / log2(f32(globals.initial_angles)); //
    let ray_spread = 0.1 / ray_scale; // must be < 0.5

    // // these are staggered to mask the edge between cascades
    let start_x = ray_scale * select(f32(start)*(1.-ray_spread), f32(start)*(1.+ray_spread), start_longer);
    let end_x = ray_scale * select(f32(end)*(1.-ray_spread), f32(end)*(1.+ray_spread), end_longer);

    return trace_radiance(probe, start_x, end_x - start_x);
}

// This is some random garbage for testing.
// It should be replaced with an actual map of density  (walls, smoke, fog, etc.)
// Location is in world coorinates.
//
// Note that this is not SDF.
// It's not exactly opacity, because it goes from 0 to infinity.
fn sample_density(location: vec2f) -> f32 {
    let tile = vec2i(location / 200.);
    let point = tile * 200 + 100;
    let dist = length(location - vec2f(point));
    let odd = (tile.x ^ tile.y) & 3;

    let obstacle_radius = (f32(odd)+0.15) * 8.;

    if dist < obstacle_radius {
        return 20. / obstacle_radius;
    }

    if location.y < 1150. && location.y > 1050. {
        return pow(0.6 * location.x / f32(globals.world_size.x), 2.);
    }
    if location.x > 300. && location.x <1220. && location.y > 400. && location.y < 400. + (location.x-300.) / 20. {
        return 10.9;
    }

    if tile.y == 3 && abs(location.x - 666.) < 10. {
        return 10.;
    }
    if tile.y == 4 && abs(location.x - 666.) < 10. {
        return 0.1;
    }

    if location.x < 300. {
        return 0.015; // smoke
    }
    return 0.0025;
}

// Random garbage for testing.
//
// Returns amount of light emitted from here, world coordinates, in linear RGB.
// The angle (radians) from the raymarch, so it's actually reverse of the light ray direction.
fn sample_light(location: vec2f, angle: f32) -> vec3f {
   let distx = length(location - vec2f(103.,53.));
   if distx < 5. {
       return vec3f(0.95);
   }

   let dist = length(location - globals.mouse_position);
   if dist < 5. {
       return vec3f(13., 0., 0.) * pow(max(0., cos(globals.time * 2. - angle)), 3.);
   }

   let diff = location - globals.mouse_position;
   if diff.x > 300. && diff.x < 500. && diff.y > 250. && diff.y < 253. {
       return vec3f(1., 1., 4.);
   }

   let dist2 = length(location - globals.mouse_position + vec2f(100.,100.));
   if dist2 < 15. {
       return vec3f(0.1, 0.7, 0.);
   }

   return vec3f(0.);
}
