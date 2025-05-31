use crate::colour::Colour;
use crate::context::WgpuClump;
use crate::engine_handle::Engine;
use crate::render::Renderer;
use crate::resource::ResourceId;
use crate::shader::{Shader, UniformData, UniformError};
use crate::texture::{Texture, UniformTexture};
use crate::vectors::Vec2;
use crate::vertex::{self, LineVertex, Vertex};

#[derive(Debug)]
pub struct batch_renderer<VertexType = (), UniformDataType = ()> {
    pipeline_id: ResourceId<Shader>,
    pub(crate) index_count: u64,
    pub(crate) vertex_count: u64,
    inner: Option<InnerBuffer>,
    texture_id: ResourceId<Texture>,
    _marker_vt: PhantomData<VertexType>,
    _marker_udt: PhantomData<UniformDataType>,
}
