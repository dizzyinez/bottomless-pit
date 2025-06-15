use std::marker::PhantomData;

use crate::colour::Colour;
use crate::graphics_context::WgpuClump;
use crate::engine_handle::Engine;
use crate::render::Renderer;
use crate::resource::ResourceId;
use crate::shader::{Shader, UniformData, UniformError};
use crate::texture::{Texture, UniformTexture};
use crate::vectors::Vec2;
use crate::vertex::{self, Batchable, Layout, LineVertex, Vertex2D};

#[derive(Debug)]
pub struct SpriteBatch<
    VertexType: Layout + Batchable + bytemuck::Pod = Vertex2D,
    UniformDataType = (),
> {
    pipeline_id: ResourceId<Shader<VertexType>>,
    //pub(crate) index_count: u64,
    //pub(crate) vertex_count: u64,
    //inner: Option<InnerBuffer>,
    //texture_id: ResourceId<Texture>,
    max_materials: u32,
    max_vertices: u16,
    max_indices: u16,
    num_vertices: u16,
    num_indices: u16,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    _marker_vt: PhantomData<VertexType>,
    _marker_udt: PhantomData<UniformDataType>,
}

impl<VertexType: Layout + Batchable + bytemuck::Pod, UniformDataType>
    SpriteBatch<VertexType, UniformDataType>
{
    const VERTEX_SIZE: u16 = std::mem::size_of::<VertexType>() as u16;
    const INDEX_SIZE: u16 = std::mem::size_of::<u16>() as u16;

    //pub fn from_shader(device: &wgpu::Device, pipeline_id: ResourceId<Shader>) -> Self {
    pub fn from_shader(engine: &Engine, pipeline_id: ResourceId<Shader<VertexType>>) -> Self {
        let device = &engine.graphics_context.as_ref().unwrap().wgpu.device;
        let (vertex_buffer, index_buffer) =
            Self::create_buffers(device, 4096, 8192);
        Self {
            pipeline_id,
            max_materials: device.limits().max_binding_array_elements_per_shader_stage,
            max_vertices: vertex_buffer.size() as u16 / Self::VERTEX_SIZE,
            max_indices: index_buffer.size() as u16 / Self::INDEX_SIZE,
            num_vertices: 0,
            num_indices: 0,
            vertex_buffer,
            index_buffer,
            _marker_vt: PhantomData,
            _marker_udt: PhantomData,
        }
    }

    pub(crate) fn create_buffers(
        device: &wgpu::Device,
        max_vertices: u16,
        max_indices: u16,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex_Buffer"),
            size: Self::VERTEX_SIZE as u64 * max_vertices as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index_Buffer"),
            size: Self::INDEX_SIZE as u64 * max_indices as u64,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        (vertex_buffer, index_buffer)
    }

    /// Draws all queued shapes to the screen.
    pub fn flush(&mut self, renderer: &mut Renderer<'_, '_>) {
        if self.num_vertices == 0 {
            return;
        }

        // returns early bc stuff isnt loaded so we just ignore it ! :3
        let Some(shader) = renderer.resources.get_pipeline(&self.pipeline_id) else {
            return;
        };

        //let Some(texture) = renderer
        //    .resources
        //    .get_texture(&self.texture_id)
        //    .map(|t| &t.bind_group)
        //else {
        //    return;
        //};
        shader.set_active(renderer);

        //renderer.pass.set_bind_group(0, texture, &[]);

        // should never panic as the vertex == 0 means that there has been
        // some data put in which means this should be Some(T)
        renderer.pass.set_vertex_buffer(
            0,
            self.vertex_buffer
                .slice(0..(self.num_vertices as u64 * Self::VERTEX_SIZE as u64)),
        );
        renderer.pass.set_index_buffer(
            self.index_buffer
                .slice(0..(self.num_indices as u64 * Self::INDEX_SIZE as u64)),
            wgpu::IndexFormat::Uint16,
        );

        renderer
            .pass
            .draw_indexed(0..self.num_indices as u32, 0, 0..1);

        self.num_vertices = 0;
        self.num_indices = 0;
    }

    pub fn add_shape(
        &mut self,
        renderer: &mut Renderer<'_, '_>,
        vertices: &[VertexType],
        indices: &[u16],
    ) {
        if self.num_vertices + vertices.len() as u16 > self.max_vertices
            || self.num_indices + indices.len() as u16 > self.max_indices
        {
            //let &'pass pass_self = self;
            self.flush(renderer);
        }
        //let offset: wgpu::BufferAddress = self.num_vertices * Self::VERTEX_SIZE;
        //let vertex_buffer = &self.vertex_buffer.unwrap();
        renderer.wgpu.queue.write_buffer(
            &self.vertex_buffer,
            self.num_vertices as u64 * Self::VERTEX_SIZE as u64,
            //TODO: optimize heap allocations by using a short vector or by making modifications
            //directly in the memory allocated by the queue
            bytemuck::cast_slice(
                &vertices
                    .iter()
                    .map(|&v| {
                        let mut v = v;
                        v.set_material_index(0);
                        v
                    })
                    .collect::<Vec<VertexType>>()[..],
            ),
        );

        renderer.wgpu.queue.write_buffer(
            &self.index_buffer,
            self.num_indices as u64 * Self::INDEX_SIZE as u64,
            //TODO: optimize heap allocations by using a short vector or by making modifications
            //directly in the memory allocated by the queue
            bytemuck::cast_slice(
                &indices
                    .iter()
                    .map(|&i| i + self.num_indices)
                    .collect::<Vec<u16>>()[..],
            ),
        );

        self.num_vertices += vertices.len() as u16;
        self.num_indices += indices.len() as u16;
    }

    //fn push_rectangle(&mut self, wgpu: &WgpuClump, verts: [Vertex; 4]) {
    //    if self.inner.is_none() {
    //        let (vert, ind) =
    //            Self::create_buffers(&wgpu.device, self.vertex_size, 50, self.index_size, 50);
    //        self.inner = Some(InnerBuffer {
    //            vertex_buffer: vert,
    //            index_buffer: ind,
    //        });
    //    }

    //    let num_verts = self.get_vertex_number() as u16;
    //    let buffers = self.inner.as_mut().unwrap();

    //    let max_verts = buffers.vertex_buffer.size();
    //    if self.vertex_count + (4 * self.vertex_size) > max_verts {
    //        grow_buffer(
    //            &mut buffers.vertex_buffer,
    //            wgpu,
    //            1,
    //            wgpu::BufferUsages::VERTEX,
    //        );
    //    }

    //    let indices = [
    //        num_verts,
    //        1 + num_verts,
    //        2 + num_verts,
    //        3 + num_verts,
    //        num_verts,
    //        2 + num_verts,
    //    ];

    //    let max_indices = buffers.index_buffer.size();
    //    if self.index_count + (6 * self.index_size) > max_indices {
    //        grow_buffer(
    //            &mut buffers.index_buffer,
    //            wgpu,
    //            1,
    //            wgpu::BufferUsages::INDEX,
    //        );
    //    }

    //    wgpu.queue.write_buffer(
    //        &buffers.vertex_buffer,
    //        self.vertex_count,
    //        bytemuck::cast_slice(&verts),
    //    );
    //    wgpu.queue.write_buffer(
    //        &buffers.index_buffer,
    //        self.index_count,
    //        bytemuck::cast_slice(&indices),
    //    );

    //    self.vertex_count += 4 * self.vertex_size;
    //    self.index_count += 6 * self.index_size;
    //}

    ///// Will queue a Rectangle to be draw.
    //pub fn add_rectangle(
    //    &mut self,
    //    position: Vec2<f32>,
    //    size: Vec2<f32>,
    //    colour: Colour,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;
    //    let verts = vertex::from_pixels(position, size, colour.as_raw());

    //    self.push_rectangle(wgpu, verts);
    //}

    ///// Queues a rectangle using WGSL coordinate space. (0, 0) is the center of the screen and (-1, 1) is the top left corner
    //pub fn add_screenspace_rectangle(
    //    &mut self,
    //    position: Vec2<f32>,
    //    size: Vec2<f32>,
    //    colour: Colour,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;
    //    let screen_size = render.size;

    //    let verts = vertex::new(position, size, colour.as_raw(), screen_size);
    //    self.push_rectangle(wgpu, verts);
    //}

    ///// Queues a rectangle with UV coordniates. The position and size of the UV coordinates are the same as the pixels in the
    ///// actual image.
    //pub fn add_rectangle_with_uv(
    //    &mut self,
    //    position: Vec2<f32>,
    //    size: Vec2<f32>,
    //    uv_position: Vec2<f32>,
    //    uv_size: Vec2<f32>,
    //    colour: Colour,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;

    //    let texture_size = render
    //        .resources
    //        .get_texture(&self.texture_id)
    //        .map(|t| t.size)
    //        .unwrap_or(Vec2 { x: 1.0, y: 1.0 });
    //    // doesnt matter what i put here bc the texture isnt loaded regardless

    //    let uv_size = uv_size / texture_size;
    //    let uv_position = uv_position / texture_size;

    //    let verts =
    //        vertex::from_pixels_with_uv(position, size, colour.as_raw(), uv_position, uv_size);

    //    self.push_rectangle(wgpu, verts);
    //}

    ///// Queues a rectangle that will be rotated around its centerpoint. Rotation is in degrees
    //pub fn add_rectangle_with_rotation(
    //    &mut self,
    //    position: Vec2<f32>,
    //    size: Vec2<f32>,
    //    colour: Colour,
    //    rotation: f32,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;

    //    let verts = vertex::from_pixels_with_rotation(position, size, colour.as_raw(), rotation);

    //    self.push_rectangle(wgpu, verts);
    //}

    //#[allow(clippy::too_many_arguments)]
    ///// Queues a rectangle with both UV, and Rotation,
    //pub fn add_rectangle_ex(
    //    &mut self,
    //    position: Vec2<f32>,
    //    size: Vec2<f32>,
    //    colour: Colour,
    //    rotation: f32,
    //    uv_position: Vec2<f32>,
    //    uv_size: Vec2<f32>,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;

    //    let texture_size = render.resources.get_texture(&self.texture_id).unwrap().size;

    //    let uv_size = uv_size / texture_size;
    //    let uv_position = uv_position / texture_size;

    //    let verts = vertex::from_pixels_ex(
    //        position,
    //        size,
    //        colour.as_raw(),
    //        rotation,
    //        uv_position,
    //        uv_size,
    //    );

    //    self.push_rectangle(wgpu, verts);
    //}

    //#[allow(clippy::too_many_arguments)]
    ///// Queues a rectangle with both UV, and Rotation, but will draw the rectangle in WGSL screenspace
    //pub fn add_screenspace_rectangle_ex(
    //    &mut self,
    //    position: Vec2<f32>,
    //    size: Vec2<f32>,
    //    colour: Colour,
    //    rotation: f32,
    //    uv_position: Vec2<f32>,
    //    uv_size: Vec2<f32>,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;

    //    let verts = vertex::new_ex(
    //        position,
    //        size,
    //        colour.as_raw(),
    //        rotation,
    //        uv_position,
    //        uv_size,
    //    );

    //    self.push_rectangle(wgpu, verts);
    //}

    ///// Queues a 4 pointed polygon with complete control over uv coordinates and rotation. The points need to be in top left, right
    ///// bottom right and bottom left order as it will not render porperly otherwise.
    //pub fn add_custom(
    //    &mut self,
    //    points: [Vec2<f32>; 4],
    //    uv_points: [Vec2<f32>; 4],
    //    rotation: f32,
    //    colour: Colour,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;
    //    let texture_size = render.resources.get_texture(&self.texture_id).unwrap().size;
    //    let uv_points = [
    //        uv_points[0] / texture_size,
    //        uv_points[1] / texture_size,
    //        uv_points[2] / texture_size,
    //        uv_points[3] / texture_size,
    //    ];

    //    let verts = vertex::from_pixels_custom(points, uv_points, rotation, colour.as_raw());

    //    self.push_rectangle(wgpu, verts);
    //}

    ///// Queues a triangle, the points must be provided in clockwise order
    //pub fn add_triangle(
    //    &mut self,
    //    p1: Vec2<f32>,
    //    p2: Vec2<f32>,
    //    p3: Vec2<f32>,
    //    colour: Colour,
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;

    //    let colour = colour.as_raw();
    //    let tex_coords = [0.0, 0.0];

    //    let verts = [
    //        Vertex::from_2d([p1.x, p1.y], tex_coords, colour),
    //        Vertex::from_2d([p2.x, p2.y], tex_coords, colour),
    //        Vertex::from_2d([p3.x, p3.y], tex_coords, colour),
    //    ];

    //    self.push_triangle(wgpu, verts);
    //}

    ///// Queues a triangle where each vertex is given its own colour. Points must be given
    ///// in clockwise order
    //pub fn add_triangle_with_coloured_verticies(
    //    &mut self,
    //    points: [Vec2<f32>; 3],
    //    colours: [Colour; 3],
    //    render: &Renderer,
    //) {
    //    let wgpu = render.wgpu;

    //    let tex_coords = [0.0, 0.0];
    //    let verts = [
    //        Vertex::from_2d([points[0].x, points[0].y], tex_coords, colours[0].as_raw()),
    //        Vertex::from_2d([points[1].x, points[1].y], tex_coords, colours[1].as_raw()),
    //        Vertex::from_2d([points[2].x, points[2].y], tex_coords, colours[2].as_raw()),
    //    ];

    //    self.push_triangle(wgpu, verts);
    //}

    ///// Queues a polygon with the specified number of sides at a position with size and colour.
    ///// This will not play nicely with texture as all the UV coords will be at [0, 0].
    //pub fn add_regular_n_gon(
    //    &mut self,
    //    number_of_sides: usize,
    //    radius: f32,
    //    center: Vec2<f32>,
    //    colour: Colour,
    //    render: &Renderer,
    //) {
    //    if number_of_sides < 4 {
    //        return;
    //    }

    //    if self.inner.is_none() {
    //        let (vert, ind) = Self::create_buffers(
    //            &render.wgpu.device,
    //            Self::VERTEX_SIZE,
    //            50,
    //            Self::INDEX_SIZE,
    //            50,
    //        );

    //        self.inner = Some(InnerBuffer {
    //            vertex_buffer: vert,
    //            index_buffer: ind,
    //        });
    //    }

    //    let wgpu = render.wgpu;

    //    let vertices = (0..number_of_sides)
    //        .map(|num| Vec2 {
    //            x: radius * (2.0 * PI * num as f32 / number_of_sides as f32).cos() + center.x,
    //            y: radius * (2.0 * PI * num as f32 / number_of_sides as f32).sin() + center.y,
    //        })
    //        .map(|point| Vertex::from_2d([point.x, point.y], [0.0, 0.0], colour.as_raw()))
    //        .collect::<Vec<Vertex>>();

    //    let number_of_vertices = self.get_vertex_number() as u16;
    //    let number_of_triangles = (number_of_sides - 2) as u16;

    //    let mut indices = (1..number_of_triangles + 1)
    //        .flat_map(|i| {
    //            [
    //                number_of_vertices,
    //                i + number_of_vertices,
    //                i + 1 + number_of_vertices,
    //            ]
    //        })
    //        .collect::<Vec<u16>>();

    //    // ensures we follow copy buffer alignment
    //    let num_indices = indices.len();
    //    let triangles_to_add = if num_indices < 12 {
    //        (12 % num_indices) / 3
    //    } else {
    //        (num_indices % 12) / 3
    //    };

    //    for _ in 0..triangles_to_add {
    //        indices.extend_from_slice(&[
    //            indices[num_indices - 3],
    //            indices[num_indices - 2],
    //            indices[num_indices - 1],
    //        ]);
    //    }

    //    let buffers = self.inner.as_mut().unwrap();

    //    let max_verts = buffers.vertex_buffer.size();
    //    if self.vertex_count + (vertices.len() as u64 * Self::VERTEX_SIZE) > max_verts {
    //        grow_buffer(
    //            &mut buffers.vertex_buffer,
    //            wgpu,
    //            self.vertex_count + (vertices.len() as u64 * Self::VERTEX_SIZE),
    //            wgpu::BufferUsages::VERTEX,
    //        );
    //    }

    //    let max_indices = buffers.index_buffer.size();
    //    if self.index_count + (indices.len() as u64 * self.index_size) > max_indices {
    //        grow_buffer(
    //            &mut buffers.index_buffer,
    //            wgpu,
    //            self.index_count + (indices.len() as u64 * self.index_size),
    //            wgpu::BufferUsages::INDEX,
    //        );
    //    }

    //    wgpu.queue.write_buffer(
    //        &buffers.vertex_buffer,
    //        self.vertex_count,
    //        bytemuck::cast_slice(&vertices),
    //    );
    //    wgpu.queue.write_buffer(
    //        &buffers.index_buffer,
    //        self.index_count,
    //        bytemuck::cast_slice(&indices),
    //    );

    //    self.vertex_count += vertices.len() as u64 * self.vertex_size;
    //    self.index_count += indices.len() as u64 * self.index_size;
    //}

    //fn push_rectangle(&mut self, wgpu: &WgpuClump, verts: [Vertex; 4]) {
    //    if self.inner.is_none() {
    //        let (vert, ind) =
    //            Self::create_buffers(&wgpu.device, self.vertex_size, 50, self.index_size, 50);
    //        self.inner = Some(InnerBuffer {
    //            vertex_buffer: vert,
    //            index_buffer: ind,
    //        });
    //    }

    //    let num_verts = self.get_vertex_number() as u16;
    //    let buffers = self.inner.as_mut().unwrap();

    //    let max_verts = buffers.vertex_buffer.size();
    //    if self.vertex_count + (4 * self.vertex_size) > max_verts {
    //        grow_buffer(
    //            &mut buffers.vertex_buffer,
    //            wgpu,
    //            1,
    //            wgpu::BufferUsages::VERTEX,
    //        );
    //    }

    //    let indices = [
    //        num_verts,
    //        1 + num_verts,
    //        2 + num_verts,
    //        3 + num_verts,
    //        num_verts,
    //        2 + num_verts,
    //    ];

    //    let max_indices = buffers.index_buffer.size();
    //    if self.index_count + (6 * self.index_size) > max_indices {
    //        grow_buffer(
    //            &mut buffers.index_buffer,
    //            wgpu,
    //            1,
    //            wgpu::BufferUsages::INDEX,
    //        );
    //    }

    //    wgpu.queue.write_buffer(
    //        &buffers.vertex_buffer,
    //        self.vertex_count,
    //        bytemuck::cast_slice(&verts),
    //    );
    //    wgpu.queue.write_buffer(
    //        &buffers.index_buffer,
    //        self.index_count,
    //        bytemuck::cast_slice(&indices),
    //    );

    //    self.vertex_count += 4 * self.vertex_size;
    //    self.index_count += 6 * self.index_size;
    //}

    //fn push_triangle(&mut self, wgpu: &WgpuClump, verts: [Vertex; 3]) {
    //    if self.inner.is_none() {
    //        let (vert, ind) =
    //            Self::create_buffers(&wgpu.device, self.vertex_size, 50, self.index_size, 50);
    //        self.inner = Some(InnerBuffer {
    //            vertex_buffer: vert,
    //            index_buffer: ind,
    //        });
    //    }

    //    let num_verts = self.get_vertex_number() as u16;
    //    let buffers = self.inner.as_mut().unwrap();

    //    let max_verts = buffers.vertex_buffer.size();
    //    if self.vertex_count + (3 * self.vertex_size) > max_verts {
    //        grow_buffer(
    //            &mut buffers.vertex_buffer,
    //            wgpu,
    //            1,
    //            wgpu::BufferUsages::VERTEX,
    //        );
    //    }

    //    // yes its wastefull to do this but this is the only way to not have
    //    // it mess up other drawings while also allowing triangles
    //    let indices = [
    //        num_verts,
    //        1 + num_verts,
    //        2 + num_verts,
    //        num_verts,
    //        1 + num_verts,
    //        2 + num_verts,
    //    ];

    //    let max_indices = buffers.index_buffer.size();
    //    if self.index_count + (6 * self.index_size) > max_indices {
    //        grow_buffer(
    //            &mut buffers.index_buffer,
    //            wgpu,
    //            1,
    //            wgpu::BufferUsages::INDEX,
    //        );
    //    }

    //    wgpu.queue.write_buffer(
    //        &buffers.vertex_buffer,
    //        self.vertex_count,
    //        bytemuck::cast_slice(&verts),
    //    );
    //    wgpu.queue.write_buffer(
    //        &buffers.index_buffer,
    //        self.index_count,
    //        bytemuck::cast_slice(&indices),
    //    );

    //    self.vertex_count += 3 * self.vertex_size;
    //    self.index_count += 6 * self.index_size;
    //}

    //// there where 'others: 'pass notation says that 'others lives longer than 'pass
    ///// Draws all queued shapes to the screen.

    //pub(crate) fn create_buffers(
    //    device: &wgpu::Device,
    //    vertex_size: u64,
    //    vert_count: u64,
    //    index_size: u64,
    //    index_count: u64,
    //) -> (wgpu::Buffer, wgpu::Buffer) {
    //    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //        label: Some("Vertex_Buffer"),
    //        size: vertex_size * vert_count,
    //        usage: wgpu::BufferUsages::VERTEX
    //            | wgpu::BufferUsages::COPY_DST
    //            | wgpu::BufferUsages::COPY_SRC,
    //        mapped_at_creation: false,
    //    });

    //    // this is just 200 bytes pretty small
    //    let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //        label: Some("Index_Buffer"),
    //        size: index_size * index_count,
    //        usage: wgpu::BufferUsages::INDEX
    //            | wgpu::BufferUsages::COPY_DST
    //            | wgpu::BufferUsages::COPY_SRC,
    //        mapped_at_creation: false,
    //    });

    //    (vertex_buffer, index_buffer)
    //}
}

#[derive(Debug)]
pub struct BatchRendererBuilder {}

pub(crate) fn grow_buffer(
    buffer: &mut wgpu::Buffer,
    wgpu: &WgpuClump,
    size_needed: u64,
    vert_or_index: wgpu::BufferUsages,
) {
    let mut encoder = wgpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Material Buffer Grower"),
        });

    let size_needed = size_needed + (4 - (size_needed % wgpu::COPY_BUFFER_ALIGNMENT));

    let new_size = std::cmp::max(buffer.size() * 2, size_needed);
    let new_buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex_Buffer"),
        size: new_size,
        usage: vert_or_index | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &new_buffer, 0, buffer.size());

    wgpu.queue.submit(std::iter::once(encoder.finish()));

    *buffer = new_buffer;
}
