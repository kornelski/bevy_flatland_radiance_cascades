use bevy::prelude::*;
use bevy::render;
use bevy::sprite::MeshMaterial2d;

const WINDOW_SIZE: UVec2 = UVec2::new(1920, 1080);

mod cascades;
use cascades::*;

mod vis;
use vis::*;

fn main() {
    App::new()
        .add_systems(Startup, init)
        .add_systems(Update, auto_init_vis_buffers)
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (
                            WINDOW_SIZE.x as f32,
                            WINDOW_SIZE.y as f32,
                        ).into(),
                        present_mode: bevy::window::PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                }),

            // Stuff is happening here:
            CascadesComputePlugin,

            QuickAndDirtyBufferPreviewPlugin,
        ))
        .run();
}

fn init(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn(CascadesSettingsComponent {
        size: WINDOW_SIZE,
    });
}


// This is just for preview in the window
#[derive(Component)]
struct QuickAndDirtyPreviewAddedMarker;

fn auto_init_vis_buffers(mut commands: Commands,
    mut materials: ResMut<Assets<PreviewMaterial>>,
    buffers: Query<(Entity, &CascadesBuffersComponent), Without<QuickAndDirtyPreviewAddedMarker>>) {

    for (entity, buffers) in &buffers {
        commands.entity(entity).insert(QuickAndDirtyPreviewAddedMarker);

        commands.spawn(QuickAndDirtyBufferPreviewSettings { x: 0, y: 0 })
            .insert(MeshMaterial2d(materials.add(PreviewMaterial {
                // debug view of merged cascades
                texture: buffers.buffers[1].clone_weak(),
                mode: 1,
            })));

        commands.spawn(QuickAndDirtyBufferPreviewSettings { x: 1, y: 0 })
            .insert(MeshMaterial2d(materials.add(PreviewMaterial {
                // debug view of merged cascades
                texture: buffers.buffers[0].clone_weak(),
                mode: 0,
            })));
    }
}
