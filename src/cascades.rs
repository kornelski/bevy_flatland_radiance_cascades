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

/// 1 = double rays on every cascade
/// 2 = quadruple rays on every cascade
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
    /// Buffers for merging
    group1_layout: BindGroupLayout,

    /// The GPU buffer that stores the [`CascadesSettingsUniform`] data.
    settings_shared_uniforms_buffer: DynamicUniformBuffer<CascadesSettingsUniform>,

    /// Configuring each cascade
    params_shared_template: Vec<CascadesParamsUniform>,
    /// copying from template to GPU
    params_shared_uniforms_buffer: DynamicUniformBuffer<CascadesParamsUniform>,
    params_shared_uniforms_buffer_offsets: [u32; NUM_CASCADES],

    /// Merge N+1 into N
    pipeline_c1: CachedComputePipelineId,
    /// Cascade 0 gets special treatment
    pipeline_c0: CachedComputePipelineId,
    /// Cascade 0 gets special treatment
    pipeline_cmax: CachedComputePipelineId,
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

        let group1_layout = render_device.create_bind_group_layout("g1",
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
            )),
        );

        // naga_oil preprocessor
        let shader_defs = vec![
            ShaderDefVal::UInt("BRANCHING_FACTOR".into(), BRANCHING_FACTOR),
        ];

        let shader = world.load_asset("radiance_cascades.wgsl");
        let pipeline_c0 = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cascades_c0".into()),
            entry_point: "cascades_c0".into(),
            layout: vec![group0_layout.clone(), group1_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
        });
        let pipeline_c1 = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cascades_c1".into()),
            entry_point: "cascades_c1".into(),
            layout: vec![group0_layout.clone(), group1_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
        });
        let pipeline_cmax = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cascades_cmax".into()),
            entry_point: "cascades_cmax".into(),
            layout: vec![group0_layout.clone(), group1_layout.clone()],
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
            group1_layout,
            pipeline_c1,
            pipeline_cmax,
            pipeline_c0,
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
        let buffer_size = (0..NUM_CASCADES as u32).map(|c| {
            let m = buffer_sizes(c, settings);
            let size = m.num_probes * m.num_angles_sqrt;
            eprintln!("buffer for c{c}; {:?} probes * {}^2 = {size:?}", m.num_probes, m.num_angles_sqrt);
            size
        }).fold(UVec2::new(0,0), |max, m| {
            max.max(m)
        });

        // one of these could be smaller for cascade 1+ only.
        let buffers = array::from_fn(|_| {
            let mut img = Image::new(
                Extent3d {
                    width: buffer_size.x,
                    height: buffer_size.y,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                vec![0; (16 * buffer_size.x * buffer_size.y) as usize],
                TextureFormat::Rgba32Float,
                RenderAssetUsages::RENDER_WORLD,
            );
            // texture binding is for a preview only
            img.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
            images.add(img)
        });

        commands.entity(entity).insert(CascadesBuffersComponent {
            buffers,
        });
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
        let [a, b] = buffers.buffers.each_ref().map(|b| gpu_images.get(b).unwrap());
        let a_bind_group = render_device.create_bind_group("a", &pipeline.group1_layout, &BindGroupEntries::sequential((
            &a.texture_view,
            &b.texture_view,
        )));
        let b_bind_group = render_device.create_bind_group("b", &pipeline.group1_layout, &BindGroupEntries::sequential((
            &b.texture_view,
            &a.texture_view,
        )));
        commands.entity(e).insert(CascadesRenderImagesBindGroupsComponent {
            bind_groups: [a_bind_group, b_bind_group],
        });
    }
}

#[derive(Debug, Component, ExtractComponent, Clone)]
pub struct CascadesBuffersComponent {
    /// for gathering merged cascades, direction-first
    pub buffers: [Handle<Image>; 2],
}

#[derive(Component)]
struct CascadesRenderImagesBindGroupsComponent {
    bind_groups: [BindGroup; 2],
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
            for p in [pipeline.pipeline_c1, pipeline.pipeline_c0] {
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

        let pipeline_cmax = pipeline_cache.get_compute_pipeline(pipeline.pipeline_cmax).unwrap();
        let pipeline_c1 = pipeline_cache.get_compute_pipeline(pipeline.pipeline_c1).unwrap();
        let pipeline_c0 = pipeline_cache.get_compute_pipeline(pipeline.pipeline_c0).unwrap();

        let mut pass = render_context.command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("cascades"),
                timestamp_writes: None,
            });

        for (buffers_bind, uniform_offset, settings) in self.view_query.iter_manual(render_world) {
            let mut first_merge = true;
            for (params, params_offset) in pipeline.params_shared_template.iter().zip(pipeline.params_shared_uniforms_buffer_offsets).rev() {
                pass.set_pipeline(if first_merge { first_merge = false; pipeline_cmax } else { pipeline_c1 });
                pass.set_bind_group(0, &settings_bind, &[uniform_offset.0, params_offset]);
                pass.set_bind_group(1, &buffers_bind.bind_groups[((params.cascade) & 1) as usize] , &[]);
                let msize = buffer_sizes(params.cascade, settings);
                let dispatch = (msize.num_probes * msize.num_angles_sqrt + (WORKGROUP_SIZE-1)) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
            }

            pass.set_pipeline(pipeline_c0);
            pass.set_bind_group(1, &buffers_bind.bind_groups[1], &[]);
            let msize = buffer_sizes(0, settings);
            let dispatch = (msize.num_probes * msize.num_angles_sqrt + (WORKGROUP_SIZE-1)) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }

        Ok(())
    }
}

struct BufferSizes {
    num_angles_sqrt: u32,
    num_probes: UVec2,
}

fn buffer_sizes(cascade: u32, settings: &CascadesSettingsComponent) -> BufferSizes {
    let num_angles_sqrt = ((INITIAL_ANGLES << (cascade * BRANCHING_FACTOR)) as f32).sqrt().ceil() as u32;
    let num_probes = (settings.size / INITIAL_SPACING) >> cascade;
        // + (3 + cascade * 3); // hacky hack
    BufferSizes {
        num_angles_sqrt,
        num_probes,
    }
}
