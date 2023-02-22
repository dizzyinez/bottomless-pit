use wgpu::util::DeviceExt;

use crate::{Vertex, Texture, cache::TextureIndex};

pub const RECT_INDICIES: &[u16] = &[
    0, 1, 2,
    3, 0, 2,
];

pub struct Rectangle {
    points: [Vertex; 4],
}

impl Rectangle {
    pub fn new(pos: [f32; 2], size: [f32; 2], colour: [f32; 4]) -> Self {
        let points = [
            Vertex::from_2d(pos, [0.0, 0.0], colour), 
            Vertex::from_2d([pos[0] + size[0], pos[1]], [1.0, 0.0], colour),
            Vertex::from_2d([pos[0] + size[0], pos[1] - size[1]], [1.0, 1.0], colour),
            Vertex::from_2d([pos[0], pos[1] - size[1]], [0.0, 1.0], colour),
        ];

        Self {
            points,
        }
    }

    pub fn from_points(points: [[f32; 2]; 4], colour: [f32; 4]) -> Self {
        let points = [
            Vertex::from_2d(points[0], [0.0, 0.0], colour),
            Vertex::from_2d(points[1], [1.0, 0.0], colour),
            Vertex::from_2d(points[2], [1.0, 1.0], colour),
            Vertex::from_2d(points[3], [0.0, 1.0], colour),
        ]; 

        Self {
            points,
        }
    }

    pub fn from_corners(corners: [[f32; 2]; 2], colour: [f32; 4], device: &wgpu::Device) -> Self {
        let width = corners[0][0] - corners[1][0];
        let height = corners[1][1] - corners[0][1];
        
        Self::new(corners[0], [width, height], colour)
    }

    pub fn get_width(&self) -> f32 {
        let left = self.points[0].get_pos_2d()[0];
        let right = self.points[1].get_pos_2d()[0];

        if left < 0.0 && right > 0.0 {
            left.abs() + right
        } else {
            (right - left).abs()
        }
    }

    pub fn get_height(&self) -> f32 {
        let top = self.points[0].get_pos_2d()[1];
        let bottom = self.points[2].get_pos_2d()[1];

        if top < 0.0 && bottom > 0.0 {
            top.abs() + bottom
        } else {
            (bottom - top).abs()
        }
    }

    pub fn get_vertices(&self) -> [Vertex; 4] {
        self.points
    }
}

pub struct TexturedRect {
    pub texture: TextureIndex,
    points: [Vertex; 4],
    vertex_buffer: wgpu::Buffer,
}

impl TexturedRect {
    pub fn new(texture: TextureIndex, pos: [f32; 2], size: [f32; 2], device: &wgpu::Device) -> Self {
        let colour = [1.0, 1.0, 1.0, 1.0];
        let points = [
            Vertex::from_2d(pos, [0.0, 0.0], colour), 
            Vertex::from_2d([pos[0] + size[0], pos[1]], [1.0, 0.0], colour),
            Vertex::from_2d([pos[0] + size[0], pos[1] - size[1]], [1.0, 1.0], colour),
            Vertex::from_2d([pos[0], pos[1] - size[1]], [0.0, 1.0], colour),
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&points),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            texture,
            points,
            vertex_buffer,
        }
    }

    pub fn get_texture_id(&self) -> u32 {
        self.texture.id
    }

    pub(crate) fn get_texutre(&self) -> &TextureIndex {
        &self.texture
    }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.texture.bind_group_layout
    }

    pub fn get_vertices(&self) -> [Vertex; 4] {
        self.points
    }
}

// impl<'a, 'b> DrawRectangles<'b> for wgpu::RenderPass<'a> where 'b: 'a, {
//     fn draw_rectangle(&mut self, rect: &'a Rectangle, index_buffer: &'a wgpu::Buffer, white_pixel: &'a wgpu::BindGroup, camera_bind_group: &'a wgpu::BindGroup) {
//         self.set_vertex_buffer(0, rect.vertex_buffer.slice(..));
//         self.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
//         self.set_bind_group(0, white_pixel, &[]);
//         self.set_bind_group(1, camera_bind_group, &[]);
//         self.draw_indexed(0..6, 0, 0..1);
//     }

//     fn draw_textured_rect(&mut self, rect: &'b TexturedRect, index_buffer: &'a wgpu::Buffer, camera_bind_group: &'b wgpu::BindGroup) {
//         self.set_vertex_buffer(0, rect.vertex_buffer.slice(..));
//         self.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
//         self.set_bind_group(0, &rect.texture.bind_group, &[]);
//         self.set_bind_group(1, camera_bind_group, &[]);
//         self.draw_indexed(0..6, 0, 0..1);
//     }
// }