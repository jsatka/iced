use crate::graphics::{Antialiasing, Shell};
use crate::isolated_layer;
use crate::primitive;
use crate::quad;
use crate::text;
use crate::triangle;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct Engine {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) device_epoch: u64,

    pub(crate) quad_pipeline: quad::Pipeline,
    pub(crate) text_pipeline: text::Pipeline,
    pub(crate) triangle_pipeline: triangle::Pipeline,
    #[cfg(any(feature = "image", feature = "svg"))]
    pub(crate) image_pipeline: crate::image::Pipeline,
    pub(crate) primitive_storage: Arc<RwLock<primitive::Storage>>,
    pub(crate) composite_storage: Arc<RwLock<isolated_layer::CompositeStorage>>,
    pub(crate) layer_effect_storage: Arc<RwLock<isolated_layer::LayerEffectStorage>>,
    _shell: Shell,
}

impl Engine {
    pub fn new(
        _adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        antialiasing: Option<Antialiasing>, // TODO: Initialize AA pipelines lazily
        shell: Shell,
    ) -> Self {
        static NEXT_DEVICE_EPOCH: AtomicU64 = AtomicU64::new(1);

        Self {
            format,
            device_epoch: NEXT_DEVICE_EPOCH.fetch_add(1, Ordering::Relaxed),

            quad_pipeline: quad::Pipeline::new(&device, format),
            text_pipeline: text::Pipeline::new(&device, &queue, format),
            triangle_pipeline: triangle::Pipeline::new(&device, format, antialiasing),

            #[cfg(any(feature = "image", feature = "svg"))]
            image_pipeline: {
                let backend = _adapter.get_info().backend;

                crate::image::Pipeline::new(&device, format, backend)
            },

            primitive_storage: Arc::new(RwLock::new(primitive::Storage::default())),
            composite_storage: Arc::new(RwLock::new(isolated_layer::CompositeStorage::default())),
            layer_effect_storage: Arc::new(RwLock::new(
                isolated_layer::LayerEffectStorage::default(),
            )),

            device,
            queue,
            _shell: shell,
        }
    }

    #[cfg(any(feature = "image", feature = "svg"))]
    pub fn create_image_cache(&self) -> crate::image::Cache {
        self.image_pipeline
            .create_cache(&self.device, &self.queue, &self._shell)
    }

    pub fn trim(&mut self) {
        self.text_pipeline.trim();

        // Primitive pipelines are shared by every renderer cloned from this Engine. A renderer's
        // draw boundary is not a global frame boundary, so trimming here could evict state another
        // renderer has prepared. Retain these lazily-created families for the Engine lifetime.
    }
}
