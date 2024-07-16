// Common import implemented via naga_oil used in Bevy

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

/// Just the aspect ratio of the cascades' internal layout
const PROBE_STORAGE_COLUMN_WIDTH: u32 = #{PROBE_STORAGE_COLUMN_WIDTH};
