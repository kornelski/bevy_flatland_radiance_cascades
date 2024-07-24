#import "cascades_common.wgsl"::{CascadesGlobals, CascadesParams, BRANCHING_FACTOR, PROBE_STORAGE_COLUMN_WIDTH};

@group(0) @binding(0) var<uniform> globals: CascadesGlobals;
@group(0) @binding(1) var<uniform> params: CascadesParams;

@group(1) @binding(0) var output: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var previous_frame: texture_storage_2d<rgba32float, read>;

struct Probe {
    world_location: vec2f,
    probe_index: vec2u,
    angle_index: u32,
    // theta, in radians
    angle: f32,
}

// Each angle of each probe is dispatched independently
fn probe_for_invocation_id(probe_angle_index: vec2u) -> Probe {
    var num_angles: u32;
    var stride: vec2u;
    if BRANCHING_FACTOR == 1 {
        num_angles = globals.initial_angles << (params.cascade * BRANCHING_FACTOR);
        stride = vec2(
            PROBE_STORAGE_COLUMN_WIDTH,
            num_angles / PROBE_STORAGE_COLUMN_WIDTH, // TODO: inaccurate for larger branching factors
        );
    } else {
        // buggy
        let num_angles_sqrt = globals.initial_angles << params.cascade;
        num_angles = num_angles_sqrt * num_angles_sqrt;
        stride = vec2(num_angles_sqrt);
    }

    // this is nth probe for this cascade level
    let probe_index = probe_angle_index / stride;

    // which angle of this probe we're going to trace now
    let angle_index =
        (probe_angle_index.x & (stride.x - 1)) +
        (probe_angle_index.y & (stride.y - 1)) * stride.x;

    // this may happen when dispatching for larger area
    let max_index = (globals.world_size / globals.initial_spacing) >> vec2u(params.cascade);
    if any(probe_index >= max_index) {
        return Probe(vec2(-1), vec2(0), 0, -1.);
    }

    // this may happen when dispatching for larger area
    if angle_index >= num_angles {
        return Probe(vec2(-1), vec2(0), 0, -1.);
    }

    // the spatial resolution keeps halving
    let world_spacing = f32(globals.initial_spacing << params.cascade);
    let world_location = vec2f(probe_index) * world_spacing
        + vec2f(world_spacing * 0.25); // center the probe within its square

    return Probe(world_location, probe_index, angle_index,
        index_to_angle(angle_index, num_angles));
}

const TAU: f32 = 6.283185307179;

fn index_to_angle(angle_index: u32, num_angles: u32) -> f32 {
    // + 0.5 starts cascade 0 with an X instead of +
    return (f32(angle_index) + 0.5) * (TAU / f32(num_angles));
}

// This is the previous frame
fn sample_fluence(world_location: vec2f) -> vec3f {
    let texture_scale = 1. / vec2f(globals.world_size) * vec2f(textureDimensions(previous_frame));
    let tex_location = world_location * texture_scale;
    return textureLoad(previous_frame, vec2u(tex_location)).rgb;
}

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
@compute @workgroup_size(8, 8, 1)
fn cascades_trace(@builtin(global_invocation_id) invocation_id: vec3u) {
    let storage_location = invocation_id.xy;
    let probe = probe_for_invocation_id(storage_location);

    // angle=-1 reports out of bounds. this may happen when dispatching for larger area
    if probe.angle < 0. {
        textureStore(output, storage_location, vec4f(0.));
        return;
    }

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

    var traced = trace_radiance(probe, start_x, end_x - start_x);

    textureStore(
        output,
        storage_location,
        traced,
    );
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
