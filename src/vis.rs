// This is a hacky way of dumping buffers into the window. There is nothing interesting here.
//
use bevy::input::{mouse::MouseButtonInput, ButtonState};
use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle, Mesh2dHandle};
use bevy::window::{PrimaryWindow, WindowResized};
use crate::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use crate::WINDOW_SIZE;
use std::marker::PhantomData;

const MAIN_GRID: Grid = Grid(2,2);

pub(crate) struct QuickAndDirtyBufferPreviewPlugin;
pub(crate) struct QuickAndDirtyBufferPreviewMaterial<M: Material2d>(PhantomData<fn(M)>);

impl<M: Material2d> Default for QuickAndDirtyBufferPreviewMaterial<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M: Material2d> Plugin for QuickAndDirtyBufferPreviewMaterial<M> {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, Self::add_meshes);
    }
}

impl<M: Material2d> QuickAndDirtyBufferPreviewMaterial<M> {
    fn add_meshes(mut commands: Commands, grid: Res<Grid>, added: Query<(Entity, &QuickAndDirtyBufferPreviewSettings, &Handle<M>), Without<Mesh2dHandle>>, windows: Query<&Window, With<PrimaryWindow>>, rect: Res<RectMesh>) {
        let win = windows.single();
        let w = win.resolution.width() as f32;
        let h = win.resolution.height() as f32;

        for (e, s, vis) in &added {
            commands.entity(e).insert(MaterialMesh2dBundle {
                mesh: rect.rect.clone(),
                transform: s.transform(*grid, w, h),
                material: vis.clone(),
                ..default()
            });
        }
    }
}

impl Plugin for QuickAndDirtyBufferPreviewPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<QuickAndDirtyBufferPreviewSettings>();
        app.add_plugins(Material2dPlugin::<PreviewMaterial>::default());
        app.add_systems(PreUpdate, Self::resize);
        app.insert_resource(MAIN_GRID);
        app.insert_resource::<MousePosition>(MousePosition::default());
        app.add_plugins(ExtractResourcePlugin::<MousePosition>::default());

        app.add_systems(Update, cursor_position_system);
        app.add_systems(Update, button_system);
        app.add_plugins(QuickAndDirtyBufferPreviewMaterial::<PreviewMaterial>::default());
        app.init_resource::<RectMesh>();
    }
}


#[derive(Resource, ExtractResource, Clone, Default)]
pub struct MousePosition {
    pub current_pos: Vec2,
    pub prev_pos: Vec2,
    pub button: bool,
}

fn cursor_position_system(mut mouse: ResMut<MousePosition>, grid: Res<Grid>, mut cursor_moved_events: EventReader<CursorMoved>) {
    for event in cursor_moved_events.read() {
        mouse.prev_pos = mouse.current_pos;
        mouse.current_pos = event.position / Vec2::new(WINDOW_SIZE.x as f32, WINDOW_SIZE.y as f32) * Vec2::new(grid.0 as f32, grid.1 as f32);
        break;
    }
}

fn button_system(mut mouse: ResMut<MousePosition>, mut button: EventReader<MouseButtonInput>) {
    for event in button.read() {
        mouse.button = event.state == ButtonState::Pressed;
        break;
    }
}

#[derive(Resource, Clone, Copy)]
pub struct Grid(u8, u8);

#[derive(Component, Reflect)]
pub(crate) struct QuickAndDirtyBufferPreviewSettings {
    pub x: u8,
    pub y: u8,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct PreviewMaterial {
    #[texture(0, visibility(fragment))]
    #[sampler(1, visibility(fragment), sampler_type="filtering")]
    pub texture: Handle<Image>,
    #[uniform(2, visibility(fragment))]
    pub mode: u32,
}

impl Material2d for PreviewMaterial {
    fn fragment_shader() -> ShaderRef {
        "vis.wgsl".into()
    }
}

#[derive(Resource)]
struct RectMesh {
    rect: Mesh2dHandle
}

impl FromWorld for RectMesh {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        RectMesh { rect: Mesh2dHandle(meshes.add(Rectangle::new(1., 1.))) }
    }
}

impl QuickAndDirtyBufferPreviewSettings {
    fn transform(&self, grid: Grid, w: f32, h: f32) -> Transform {
        let w_scaled = w / grid.0 as f32;
        let h_scaled = h / grid.1 as f32;
        let x = self.x as f32;
        let y = (grid.1 - 1 - self.y) as f32;
        Transform::from_xyz(w_scaled*(x+0.5) - w*0.5, h_scaled*(y+0.5) - h*0.5, 0.).with_scale((w_scaled, h_scaled, 1.).into())
    }
}

impl QuickAndDirtyBufferPreviewPlugin {
    fn resize(mut resize_reader: EventReader<WindowResized>, grid: Res<Grid>, mut sprites: Query<(&mut Transform, &QuickAndDirtyBufferPreviewSettings)>) {
        for w in resize_reader.read() {
            for (mut t, s) in sprites.iter_mut() {
                *t = s.transform(*grid, w.width, w.height);
            }
        }
    }
}
