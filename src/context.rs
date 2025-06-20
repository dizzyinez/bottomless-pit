use crate::engine_handle::{BuildError, DefaultResources, EngineBuilder};
use crate::layouts;
use crate::render::make_pipeline;
use crate::resource::ResourceManager;
use crate::shader::Shader;
use crate::text::TextRenderer;
use crate::texture::Texture;
use crate::vectors::Vec2;
use crate::vertex::LineVertex;
use crate::WHITE_PIXEL;
use std::sync::Arc;

use image::GenericImageView;
use wgpu::util::DeviceExt;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

pub(crate) struct GraphicsContext {
    pub(crate) wgpu: WgpuClump,
    pub(crate) surface: wgpu::Surface<'static>,
    pub(crate) texture_sampler: wgpu::Sampler,
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) camera_bind_group: wgpu::BindGroup,
    pub(crate) camera_buffer: wgpu::Buffer,
    pub(crate) text_renderer: TextRenderer,
    pub(crate) window: Arc<Window>,
}

impl GraphicsContext {
    pub fn from_active_loop(
        event_loop: &ActiveEventLoop,
        mut window_options: WindowOptions,
        resource_manager: &mut ResourceManager,
        resources: &DefaultResources,
    ) -> Self {
        // should never fail as we will always set it

        #[cfg(target_arch="wasm32")]
        let title = window_options.attributes.title.clone();

        let size: Vec2<u32> = window_options.attributes.inner_size.unwrap().into();

        #[cfg(target_arch="wasm32")]
        {
            use crate::engine_handle::BuildError;
            use winit::platform::web::WindowAttributesExtWebSys;
            use wasm_bindgen::JsCast;

            let document = web_sys::window()
                .ok_or(BuildError::CantGetWebWindow)
                .unwrap()
                .document()
                .ok_or(BuildError::CantGetDocument)
                .unwrap();
            let canvas = document.create_element("canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            let style = canvas.style();
            let _ = style.set_property("width", "100%");
            let _ = style.set_property("height", "100%");
            
            match document.get_element_by_id(&title) {
                Some(element) => {
                    let array = js_sys::Array::new();
                    array.push(&wasm_bindgen::JsValue::from(&canvas));
                    element.replace_with_with_node(&array).unwrap();
                }
                None => {
                    log::warn!(
                        "coudn't find desitantion <canvas> with id: {}, appending to body",
                        &title
                    );
                    canvas.set_id(&title);
                    let body = document.body().ok_or(BuildError::CantGetBody).unwrap();
                    body.append_child(&canvas).unwrap();
                }
            }

            window_options.attributes = window_options.attributes.with_canvas(Some(canvas));
        }

        let window = Arc::new(event_loop.create_window(window_options.attributes).unwrap());

        let backend = wgpu::Backends::all();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: backend,
            backend_options: wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions { gles_minor_version: wgpu::Gles3MinorVersion::Automatic, fence_behavior: wgpu::GlFenceBehavior::Normal },
                dx12: wgpu::Dx12BackendOptions { shader_compiler: wgpu::Dx12Compiler::Fxc },
                noop: wgpu::NoopBackendOptions { enable: (false) }
            },
            flags: wgpu::InstanceFlags::default(),
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }));

        let intermediate = Intermediate {
            surface,
            window,
            instance,
            size,
            presentation: window_options.presentation,
        };

        Self::from_contiuned(adapter, intermediate, resource_manager, resources)
    }

    pub(crate) fn from_contiuned(
        adapter: Result<wgpu::Adapter, wgpu::RequestAdapterError>,
        pre_made: Intermediate,
        resource_manager: &mut ResourceManager,
        resources: &DefaultResources,
    ) -> Self {
        let adapter = match adapter {
            Ok(a) => a,
            Err(e) => panic!("AHHHHHHHH no adapter: {}", e),
        };

        let limits = adapter.limits();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                label: None,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off, //TODO: allow users to pass through directory for trace
            },
        ))
        .unwrap();

        let wgpu_clump = WgpuClump { device, queue };

        let surface_capabilities = pre_made.surface.get_capabilities(&adapter);
        let surface_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_capabilities.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: pre_made.size.x,
            height: pre_made.size.y,
            present_mode: pre_made.presentation,
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        pre_made.surface.configure(&wgpu_clump.device, &config);

        let texture_sampler = wgpu_clump.device.create_sampler(&wgpu::SamplerDescriptor {
            // what to do when given cordinates outside the textures height/width
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            // what do when give less or more than 1 pixel to sample
            // linear interprelates between all of them nearest gives the closet colour
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let camera_matrix = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            pre_made.size.x as f32,
            pre_made.size.y as f32,
            0.0,
            0.0,
        ];

        let camera_buffer =
            wgpu_clump
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_matrix]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let camera_bind_group_layout = layouts::create_camera_layout(&wgpu_clump.device);

        let camera_bind_group = wgpu_clump
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
                label: Some("camera_bind_group"),
            });

        let texture_format = config.format;

        let white_pixel_image = image::load_from_memory(WHITE_PIXEL).unwrap();
        let white_pixel_rgba = white_pixel_image.to_rgba8();
        let (width, height) = white_pixel_image.dimensions();
        let white_pixel_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let white_pixel_texture = wgpu_clump.device.create_texture(&wgpu::TextureDescriptor {
            size: white_pixel_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            view_formats: &[],
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("White_Pixel"),
        });

        wgpu_clump.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &white_pixel_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &white_pixel_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            white_pixel_size,
        );

        let white_pixel_view =
            white_pixel_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let white_pixel_sampler = wgpu_clump.device.create_sampler(&wgpu::SamplerDescriptor {
            // what to do when given cordinates outside the textures height/width
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            // what do when give less or more than 1 pixel to sample
            // linear interprelates between all of them nearest gives the closet colour
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout = layouts::create_texture_layout(&wgpu_clump.device);

        let white_pixel_bind_group =
            wgpu_clump
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&white_pixel_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&white_pixel_sampler),
                        },
                    ],
                    label: Some("texture_bind_group"),
                });

        let white_pixel = Texture::new_direct(
            white_pixel_view,
            white_pixel_bind_group,
            Vec2 { x: 1.0, y: 1.0 },
        );

        let line_shader = wgpu_clump
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Line_Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line_shader.wgsl").into()),
            });

        let line_pipeline = make_pipeline(
            &wgpu_clump.device,
            wgpu::PrimitiveTopology::LineList,
            &[&camera_bind_group_layout],
            &[LineVertex::desc()],
            &line_shader,
            texture_format,
            Some("line_renderer"),
        );

        let text_renderer = TextRenderer::new(&wgpu_clump);

        let line_shader = Shader::from_pipeline(line_pipeline);
        let generic_shader = Shader::default(&wgpu_clump, texture_format);

        resource_manager.insert_pipeline(resources.line_pipeline_id, line_shader);
        resource_manager.insert_pipeline(resources.default_pipeline_id, generic_shader);

        resource_manager.insert_texture(resources.default_texture_id, white_pixel);

        Self {
            wgpu: wgpu_clump,
            window: pre_made.window,
            surface: pre_made.surface,
            texture_sampler,
            config,
            camera_bind_group,
            camera_buffer,
            text_renderer,
        }
    }

    pub(crate) fn get_texture_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub(crate) fn get_surface_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

pub(crate) struct WindowOptions {
    attributes: WindowAttributes,
    presentation: wgpu::PresentMode,
}

impl From<EngineBuilder> for WindowOptions {
    fn from(value: EngineBuilder) -> Self {
        let attributes = Window::default_attributes()
            .with_title(&value.window_title)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                value.resolution.0,
                value.resolution.1,
            ))
            .with_resizable(value.resizable)
            .with_window_icon(value.window_icon);

        let attributes = if value.full_screen {
            attributes.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        } else {
            attributes
        };

        Self {
            attributes,
            presentation: value.vsync,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Intermediate {
    surface: wgpu::Surface<'static>,
    window: Arc<Window>,
    size: Vec2<u32>,
    instance: wgpu::Instance,
    presentation: wgpu::PresentMode,
}

pub(crate) struct WgpuClump {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}
