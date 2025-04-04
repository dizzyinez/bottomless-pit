use std::f32::consts::PI;

use bottomless_pit::colour::Colour;
use bottomless_pit::engine_handle::{Engine, EngineBuilder};
use bottomless_pit::material::{Material, MaterialBuilder};
use bottomless_pit::render::RenderHandle;
use bottomless_pit::resource::LoadingOp;
use bottomless_pit::shader::UniformData;
use bottomless_pit::shader::{Shader, ShaderOptions};
use bottomless_pit::vec2;
use bottomless_pit::vectors::Vec2;
use bottomless_pit::Game;

use encase::ShaderType;

fn main() {
    let mut engine = EngineBuilder::new()
        .with_resolution((500, 500))
        .remove_vsync()
        .build()
        .unwrap();

    let data = Time {
        time: 0.0,
        _pading: 0.0,
        _padding2: 0.0,
        _padding4: 0.0,
    };

    let uniform_data = UniformData::new(&data);

    let mouse_shader = Shader::new(
        "examples/sinewaves.wgsl",
        ShaderOptions::with_uniform_data(&uniform_data),
        &mut engine,
        LoadingOp::Blocking,
    );

    let regular_material = MaterialBuilder::new()
        .set_shader(mouse_shader)
        .build(&mut engine);

    let pos = Position {
        regular_material,
        time: data,
    };

    engine.run(pos);
}

#[derive(ShaderType)]
struct Time {
    time: f32,
    _pading: f32,
    _padding2: f32,
    _padding4: f32,
}

struct Position {
    regular_material: Material<Time>,
    time: Time,
}

impl Game for Position {
    fn render<'o>(&'o mut self, mut render: RenderHandle<'o>) {
        let mut render_handle = render.begin_pass(Colour::BLACK);

        self.regular_material.add_regular_n_gon(
            120,
            200.0,
            vec2! { 250.0 },
            Colour::BLUE,
            &render_handle,
        );

        self.regular_material.draw(&mut render_handle);
    }

    fn update(&mut self, engine_handle: &mut Engine) {
        let dt = engine_handle.get_frame_delta_time();
        self.time.time = (self.time.time + dt) % (32.0 * PI);
        self.regular_material
            .update_uniform_data(&self.time, engine_handle)
            .unwrap_or(());
    }
}
