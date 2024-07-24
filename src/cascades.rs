/// Radiance cascades very vanilla implementation
///
/// (probably buggy)
///
/// - no pre-averaging yet
/// - no bilinear fix
/// - no mipmaps
/// - no SDFs even (naively marches density map)
/// - no proper bounce light (can this even work without SDFs?)

use std::array;
use crate::render::extract_component::ExtractComponentPlugin;
use crate::MousePosition;
use crate::render::renderer::RenderQueue;
use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::binding_types::{texture_storage_2d, uniform_buffer};
use bevy::render::{self, RenderSet};
use bevy::render::texture::GpuImage;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::render_resource::*;
use bevy::render::render_graph::{self, RenderGraph, RenderGraphContext, RenderLabel};
use bevy::render::render_asset::{RenderAssetUsages, RenderAssets};

/// The algorithm is very sensitive to these settings.

/// Number of cascades to generate. Limits max length of the rays.
///
/// With BRANCHING_FACTOR == 1 cascades use little memory.
const NUM_CASCADES: usize = 6;
/// 4, 8, 16 (must be 2^n),
const INITIAL_ANGLES: u32 = 16;
/// 1 or 2 or 4. It always halves after that
const INITIAL_SPACING: u32 = 2;

/// Just the aspect ratio of the cascades' internal layout,
/// does not affect visuals in any way.
/// Must be <= INITIAL_ANGLES
const PROBE_STORAGE_COLUMN_WIDTH: u32 = 4;

/// 1 = double rays on every cascade
/// 2 = NOT IMPLEMENTED YET. quadruple rays on every cascade
const BRANCHING_FACTOR: u32 = 1;


/// Must match the shaders
const WORKGROUP_SIZE: u32 = 8;

#[derive(Clone, Copy, Component, ExtractComponent, Debug)]
pub(crate) struct CascadesSettingsComponent {
    // Image size (probe resolution is this divided by INITIAL_SPACING)
    //
    // It'd better be divisible by 32 or so, because I'm not handling
    // rounding or bounds checking in the shaders.
    pub size: UVec2,
}

/// GPU version of CascadesSettings + extra properties
///
/// This must be kept in sync with the struct in the shaders
#[derive(ShaderType)]
struct CascadesSettingsUniform {
    /// number of cascades to compute (4..6 makes sense)
    num_cascades: u32,
    /// number of rays in the cascade 0
    initial_angles: u32,
    /// number of pixels per probe in cascade 0
    initial_spacing: u32,
    /// resolution
    world_size: UVec2,
    /// resets every 1h
    time: f32,
    /// Frame time
    delta_time: f32,
    /// if pressed
    mouse_button: u32,
    /// In screen space?
    mouse_position: Vec2,
}

/// Specifies the offset within the [`CascadesUniformBuffer`] of the
/// [`CascadesSettingsUniform`] for a specific instance.
///
/// It supports multiple independent radiance cascade simulations set up at the same time
#[derive(Component)]
pub struct CascadesRenderSettingsUniformOffsetComponent(u32);

/// Fed into uniforms. It was meant to (pre)compute more of the cascade properties,
/// but the shaders do that for now.
#[derive(ShaderType)]
struct CascadesParamsUniform {
    cascade: u32,
    steps: u32,
}

#[derive(Resource)]
struct CascadesRenderPipeline {
    /// Uniforms
    group0_layout: BindGroupLayout,
    /// Buffers for ray marching
    trace_group_layout: BindGroupLayout,
    /// Buffers for merging
    merge_group_layout: BindGroupLayout,

    /// The GPU buffer that stores the [`CascadesSettingsUniform`] data.
    settings_shared_uniforms_buffer: DynamicUniformBuffer<CascadesSettingsUniform>,

    /// Configuring each cascade
    params_shared_template: Vec<CascadesParamsUniform>,
    /// copying from template to GPU
    params_shared_uniforms_buffer: DynamicUniformBuffer<CascadesParamsUniform>,
    params_shared_uniforms_buffer_offsets: [u32; NUM_CASCADES],

    trace_pipeline: CachedComputePipelineId,
    /// Merge N+1 into N
    merge_pipeline_m1: CachedComputePipelineId,
    /// Cascade 0 gets special treatment
    merge_pipeline_m0: CachedComputePipelineId,
    /// Cascade 0 gets special treatment
    merge_pipeline_first: CachedComputePipelineId,
}

pub(crate) struct CascadesComputePlugin;

impl Plugin for CascadesComputePlugin {
    fn build(&self, app: &mut App) {
        // buffer management is pretty tedious in Bevy
        app.add_plugins(ExtractComponentPlugin::<CascadesSettingsComponent>::default());
        app.add_plugins(ExtractComponentPlugin::<CascadesBuffersComponent>::default());

        // this is meant to support multiple instances, although currently barely one works
        app.add_systems(PostUpdate, attach_buffers_to_settings);

        let render_app = app.sub_app_mut(render::RenderApp);

        render_app.add_systems(
            render::Render,
            (
                render_app_prepare_cascades_settings_uniforms.in_set(RenderSet::Prepare),
                render_app_prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
            ),
        );

        let render_world = render_app.world_mut();
        let cascades_node = CascadesRenderNode {
            ready: false,
            view_query: render_world.query_filtered()
        };
        let mut render_graph = render_world.resource_mut::<RenderGraph>();
        render_graph.add_node(CascadesRenderLabel, cascades_node);
        render_graph.add_node_edge(CascadesRenderLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(render::RenderApp);
        // this is where shaders are cached
        render_app.init_resource::<CascadesRenderPipeline>();
    }
}


/// A system that collects all [`CascadesSettings`] into one GPU buffer
fn render_app_prepare_cascades_settings_uniforms(
    mut commands: Commands,
    mut pipeline: ResMut<CascadesRenderPipeline>,
    targets: Query<(Entity, &CascadesSettingsComponent)>,
    mouse_in: Res<MousePosition>,
    time: Res<Time>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if targets.is_empty() {
        return;
    }
    let pipeline = &mut *pipeline; // ResMut

    // something magically clears it?
    debug_assert!(pipeline.settings_shared_uniforms_buffer.is_empty());
    let Some(mut writer) = pipeline.settings_shared_uniforms_buffer.get_writer(
        targets.iter().len(),
        &render_device,
        &render_queue,
    ) else {
        panic!("cascades pipeline not ready");
    };

    // copy cascade settings from Bevy-land to the uniforms in shaders
    let delta_time = time.delta_seconds();
    let time = time.elapsed_seconds_wrapped();

    for (entity, cascades_settings) in targets.iter() {
        let world_size = Vec2::new(cascades_settings.size.x as f32, cascades_settings.size.y as f32);
        let mouse_button = mouse_in.button as u32;
        let mouse_position = mouse_in.current_pos.fract() * world_size;

        let offset = writer.write(&CascadesSettingsUniform {
            num_cascades: NUM_CASCADES as _,
            initial_angles: INITIAL_ANGLES,
            initial_spacing: INITIAL_SPACING,
            world_size: cascades_settings.size,
            time,
            delta_time,
            mouse_button,
            mouse_position,
        });

        commands
            .entity(entity)
            .insert(CascadesRenderSettingsUniformOffsetComponent(offset));
    }
    drop(writer);

    // this getting cleared is annoying
    debug_assert!(pipeline.params_shared_uniforms_buffer.is_empty());
    let Some(mut writer) = pipeline.params_shared_uniforms_buffer.get_writer(
        pipeline.params_shared_template.len(),
        &render_device,
        &render_queue,
    ) else {
        panic!("cascades pipeline not ready");
    };
    for (c, offset) in pipeline.params_shared_template.iter().zip(&mut pipeline.params_shared_uniforms_buffer_offsets) {
        *offset = writer.write(c);
    }
    drop(writer);
}

impl FromWorld for CascadesRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // prepare stuff for bind groups

        let group0_layout = render_device.create_bind_group_layout("g0",
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (
                uniform_buffer::<CascadesSettingsUniform>(true),
                uniform_buffer::<CascadesParamsUniform>(true),
            )),
        );

        let trace_group_layout = render_device.create_bind_group_layout("tg1",
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
            )),
        );

        let merge_group_layout = render_device.create_bind_group_layout("mg1",
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
            )),
        );

        // naga_oil preprocessor
        let shader_defs = vec![
            ShaderDefVal::UInt("PROBE_STORAGE_COLUMN_WIDTH".into(), PROBE_STORAGE_COLUMN_WIDTH),
            ShaderDefVal::UInt("BRANCHING_FACTOR".into(), BRANCHING_FACTOR),
        ];

        let trace_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cascades_trace".into()),
            entry_point: "cascades_trace".into(),
            layout: vec![group0_layout.clone(), trace_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: world.load_asset("cascades_trace.wgsl"),
            shader_defs: shader_defs.clone(),
        });

        let shader = world.load_asset("cascades_merge.wgsl");
        let merge_pipeline_m0 = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cascades_merge_m0".into()),
            entry_point: "cascades_merge_m0".into(),
            layout: vec![group0_layout.clone(), merge_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
        });
        let merge_pipeline_first = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cascades_merge_first".into()),
            entry_point: "cascades_merge_first".into(),
            layout: vec![group0_layout.clone(), merge_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
        });
        let merge_pipeline_m1 = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cascades_merge_m1".into()),
            entry_point: "cascades_merge_m1".into(),
            layout: vec![group0_layout.clone(), merge_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs,
        });

        // settings are per instance of the simulation (globals), params are per dispatch or individual cascade level
        let mut settings_shared_uniforms_buffer = DynamicUniformBuffer::default();
        settings_shared_uniforms_buffer.set_label(Some("settings"));
        let mut params_shared_uniforms_buffer = DynamicUniformBuffer::default();
        params_shared_uniforms_buffer.set_label(Some("params"));

        let params_shared_template = (0..NUM_CASCADES as u32).map(|cascade| {
            // pow 0.75 so it doesn't grow linearly with length
            let steps = ((INITIAL_ANGLES as f32) * 2f32.powf(cascade as f32) * 0.7).ceil() as u32;
            CascadesParamsUniform { cascade, steps }
        }).collect();

        CascadesRenderPipeline {
            settings_shared_uniforms_buffer,
            params_shared_template,
            params_shared_uniforms_buffer,
            params_shared_uniforms_buffer_offsets: [0; NUM_CASCADES],
            group0_layout,
            trace_group_layout,
            merge_group_layout,
            trace_pipeline,
            merge_pipeline_m1,
            merge_pipeline_first,
            merge_pipeline_m0,
        }
    }
}

/// In Bevy it's enough to create `CascadesSettingsComponent` instance,
/// and this will find it and alloc everything else for it.
/// TODO: this should use new Bevy observers?
fn attach_buffers_to_settings(mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    missing_buffers: Query<(Entity, &CascadesSettingsComponent), Without<CascadesBuffersComponent>>,
) {
    if missing_buffers.is_empty() {
        return;
    }

    let images = &mut *images; // ResMut
    for (entity, settings) in missing_buffers.iter() {
        // rest keeps same height for visualization consistency, but needs to cram into
        // variable number of columns
        let cascade_trace_size = (0..NUM_CASCADES as u32)
            .map(|c| {
                let size = cascade_storage_size(c, settings);
                eprintln!("trace buffer cascade={c}; {size:?}");
                size
            })
            .fold(UVec2::new(0,0), |max, m| max.max(m));

        let mut cascade_trace = Image::new(
            Extent3d {
                width: cascade_trace_size.x,
                height: cascade_trace_size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![0; (16 * cascade_trace_size.x * cascade_trace_size.y) as usize],
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        // texture binding is for debug only
        cascade_trace.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        let cascade_trace = images.add(cascade_trace);

        let merge_buffer_size = (0..NUM_CASCADES as u32).map(|c| {
            let m = merge_buffer_sizes(c, settings);
            let size = m.num_probes * m.num_angles_sqrt;
            eprintln!("merge buffer cascade={c}; {:?} probes * {}^2 = {size:?}", m.num_probes, m.num_angles_sqrt);
            size
        }).fold(UVec2::new(0,0), |max, m| {
            max.max(m)
        });

        // one of these could be smaller for cascade 1+ only.
        let merge = array::from_fn(|_| {
            let mut merge = Image::new(
                Extent3d {
                    width: merge_buffer_size.x,
                    height: merge_buffer_size.y,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                vec![0; (16 * merge_buffer_size.x * merge_buffer_size.y) as usize],
                TextureFormat::Rgba32Float,
                RenderAssetUsages::RENDER_WORLD,
            );
            // texture binding is for merge
            merge.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
            images.add(merge)
        });

        commands.entity(entity).insert(CascadesBuffersComponent {
            cascade_trace,
            merge,
        });
    }
}

// currentlby buggy for BRANCHING_FACTOR != 1
fn cascade_storage_size(n: u32, settings: &CascadesSettingsComponent) -> UVec2 {
    assert!(PROBE_STORAGE_COLUMN_WIDTH <= INITIAL_ANGLES);

    if n == 0 {
         (settings.size / INITIAL_SPACING) *
            UVec2::new(PROBE_STORAGE_COLUMN_WIDTH, INITIAL_ANGLES / PROBE_STORAGE_COLUMN_WIDTH)
    } else {
        let cascade_m0_height = (settings.size.y / INITIAL_SPACING) * (INITIAL_ANGLES / PROBE_STORAGE_COLUMN_WIDTH);
        debug_assert_eq!(cascade_m0_height, cascade_storage_size(0, settings).y);

        // always halving per cascade
        let probes_size = (settings.size / INITIAL_SPACING) >> n;
        debug_assert!(probes_size.x > 0 && probes_size.y > 0, "{n}");
        let angles_per_probe_column = cascade_m0_height / probes_size.y;
        debug_assert!(angles_per_probe_column > 0, "{cascade_m0_height}, {n}={probes_size:?}");
        let angles = INITIAL_ANGLES << (n * BRANCHING_FACTOR);
        let angle_columns = angles.div_ceil(angles_per_probe_column);

        UVec2::new(probes_size.x * angle_columns, cascade_m0_height)
    }
}

// Bevy has this weird thing where the normal world can't talk to the GPU,
// and the GPU part has no persistent state,
// so a bunch of things has to be redundantly recreated ;(
fn render_app_prepare_bind_groups(
    mut commands: Commands,
    gpu_images: Res<RenderAssets<GpuImage>>,
    // `Without` here sucks, because it's the render world, so it's perishable
    mut buffers: Query<(Entity, &mut CascadesBuffersComponent), Without<CascadesRenderImagesBindGroupsComponent>>,
    render_device: Res<RenderDevice>,
    pipeline: Res<CascadesRenderPipeline>,
) {
    for (e, buffers) in &mut buffers {
        let cascade_trace = gpu_images.get(&buffers.cascade_trace).unwrap();
        let [merge_a, merge_b] = buffers.merge.each_ref().map(|b| gpu_images.get(b).unwrap());

        let trace_bind_group = render_device.create_bind_group("m0", &pipeline.trace_group_layout, &BindGroupEntries::sequential((
            &cascade_trace.texture_view,
            &merge_a.texture_view,
        )));
        let merge_a_bind_group = render_device.create_bind_group("mma", &pipeline.merge_group_layout, &BindGroupEntries::sequential((
            &merge_a.texture_view,
            &cascade_trace.texture_view,
            &merge_b.texture_view,
        )));
        let merge_b_bind_group = render_device.create_bind_group("mmb", &pipeline.merge_group_layout, &BindGroupEntries::sequential((
            &merge_b.texture_view,
            &cascade_trace.texture_view,
            &merge_a.texture_view,
        )));
        commands.entity(e).insert(CascadesRenderImagesBindGroupsComponent {
            trace_bind_group,
            merge_bind_groups: [merge_a_bind_group, merge_b_bind_group],
        });
    }
}

#[derive(Debug, Component, ExtractComponent, Clone)]
pub struct CascadesBuffersComponent {
    /// Buffer for traced rays, as large as the largest cascade
    /// stored position-first
    pub cascade_trace: Handle<Image>,

    /// for gathering merged cascades,
    /// direction-first
    pub merge: [Handle<Image>; 2],
}

#[derive(Component)]
struct CascadesRenderImagesBindGroupsComponent {
    trace_bind_group: BindGroup,
    merge_bind_groups: [BindGroup; 2],
}

/// One node renders all instances, so it keeps a query for the instances
struct CascadesRenderNode {
    ready: bool,
    view_query: QueryState<(&'static CascadesRenderImagesBindGroupsComponent, &'static CascadesRenderSettingsUniformOffsetComponent, &'static CascadesSettingsComponent)>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct CascadesRenderLabel;

impl render_graph::Node for CascadesRenderNode {
    // Bevy boilerplate for loading shaders and querying world from render node
    fn update(&mut self, render_world: &mut World) {
        if !self.ready {
            let Some(pipeline) = render_world.get_resource::<CascadesRenderPipeline>() else { return };
            let pipeline_cache = render_world.resource::<PipelineCache>();
            for p in [pipeline.trace_pipeline, pipeline.merge_pipeline_m1, pipeline.merge_pipeline_m0] {
                match pipeline_cache.get_compute_pipeline_state(p) {
                    CachedPipelineState::Ok(_) => {},
                    CachedPipelineState::Err(err) => panic!("{err}"),
                    _ => {
                        return;
                    },
                }
            }
            self.ready = true;
        }
        self.view_query.update_archetypes(render_world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        render_world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !self.ready {
            return Ok(());
        }

        let pipeline = render_world.resource::<CascadesRenderPipeline>();
        let pipeline_cache = render_world.resource::<PipelineCache>();

        let render_device = render_context.render_device();

        let settings_bind = render_device.create_bind_group("settings",
            &pipeline.group0_layout,
            &BindGroupEntries::sequential((
                &pipeline.settings_shared_uniforms_buffer,
                &pipeline.params_shared_uniforms_buffer,
            )),
        );

        let trace_pipeline = pipeline_cache.get_compute_pipeline(pipeline.trace_pipeline).unwrap();
        let merge_pipeline_first = pipeline_cache.get_compute_pipeline(pipeline.merge_pipeline_first).unwrap();
        let merge_pipeline_m1 = pipeline_cache.get_compute_pipeline(pipeline.merge_pipeline_m1).unwrap();
        let merge_pipeline_m0 = pipeline_cache.get_compute_pipeline(pipeline.merge_pipeline_m0).unwrap();

        let mut pass = render_context.command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("cascades"),
                timestamp_writes: None,
            });

        for (buffers_bind, uniform_offset, settings) in self.view_query.iter_manual(render_world) {
            let mut first_merge = true;
            for (params, params_offset) in pipeline.params_shared_template.iter().zip(pipeline.params_shared_uniforms_buffer_offsets).rev() {
                pass.set_pipeline(trace_pipeline);
                pass.set_bind_group(0, &settings_bind, &[uniform_offset.0, params_offset]);
                pass.set_bind_group(1, &buffers_bind.trace_bind_group, &[]);
                let dispatch = (cascade_storage_size(params.cascade, settings)  + (WORKGROUP_SIZE-1)) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);

                pass.set_pipeline(if first_merge { first_merge = false; merge_pipeline_first } else { merge_pipeline_m1 });
                pass.set_bind_group(1, &buffers_bind.merge_bind_groups[((params.cascade) & 1) as usize] , &[]);
                let msize = merge_buffer_sizes(params.cascade, settings);
                let dispatch = (msize.num_probes * msize.num_angles_sqrt + (WORKGROUP_SIZE-1)) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
            }

            pass.set_pipeline(merge_pipeline_m0);
            pass.set_bind_group(1, &buffers_bind.merge_bind_groups[1], &[]);
            let msize = merge_buffer_sizes(0, settings);
            let dispatch = (msize.num_probes * msize.num_angles_sqrt + (WORKGROUP_SIZE-1)) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }

        Ok(())
    }
}

struct MergeBufferSizes {
    num_angles_sqrt: u32,
    num_probes: UVec2,
}

fn merge_buffer_sizes(cascade: u32, settings: &CascadesSettingsComponent) -> MergeBufferSizes {
    let num_angles_sqrt = ((INITIAL_ANGLES << (cascade * BRANCHING_FACTOR)) as f32).sqrt().ceil() as u32;
    let num_probes = ((settings.size / INITIAL_SPACING) >> cascade);
        // + (3 + cascade * 3); // hacky hack
    MergeBufferSizes {
        num_angles_sqrt,
        num_probes,
    }
}
