use crate::renderer::wgpu::core::isolated_layer::{CacheKeepAlive, CacheResidencyPriority};

use crate::core::isolated_layer::SurfaceHandle;

/// Determines which widget traversal keeps retained isolated-layer values resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheKeepAliveScope {
    /// Keeps pixels resident only while the retained producer is visibly recorded.
    #[default]
    VisibleOnly,
    /// Also marks pixels while redraw events reach the widget's update traversal.
    ///
    /// This is a best-effort policy. Event capture and widget-specific routing may prevent a
    /// mounted widget from receiving a redraw event; a visible producer still marks itself while
    /// drawing.
    KeepWhileRedrawVisited,
}

#[derive(Debug, Clone)]
pub(crate) struct CacheConfig {
    pub(crate) surface: SurfaceHandle,
    pub(crate) priority: CacheResidencyPriority,
}

impl CacheConfig {
    pub fn new(surface: &SurfaceHandle, priority: CacheResidencyPriority) -> Self {
        Self {
            surface: surface.clone(),
            priority,
        }
    }

    pub fn keep_alive(&self) -> CacheKeepAlive {
        self.surface.cache_keep_alive_with(self.priority)
    }
}
