struct BP_Vertex2D {
    @location(0) material_index: u32,
    @location(1) position: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) colour: vec4<f32>
}

struct LineVertex {
    @location(0) position: vec2<f32>,
    @location(1) colour: vec4<f32>
}

struct BP_Uniforms {
    camera: mat3x3<f32>,
    screen_size: vec2<f32>,
}
