/// gpu/mod.rs — GPU-accelerated Gaussian Splatting via wgpu
///
/// Uses compute shaders for:
///   1. Projection (parallel over Gaussians)
///   2. Prefix-sum radix sort (by depth key)
///   3. Tile rasterization (parallel over pixels)
///
/// Falls back gracefully to CPU if no GPU is available.

// `context` compiles in both configurations: it provides a real `GpuContext`
// under the `gpu` feature and a stub otherwise, so the pipeline can refer to
// `crate::gpu::GpuContext` unconditionally.
pub mod context;
pub use context::GpuContext;

#[cfg(feature = "gpu")]
pub mod compute;
#[cfg(feature = "gpu")]
pub mod sort;

#[cfg(feature = "gpu")]
pub use compute::GpuRenderer;

/// Check if a GPU is available on this system.
pub async fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        let instance = wgpu::Instance::default();
        instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }).await.is_some()
    }
    #[cfg(not(feature = "gpu"))]
    false
}