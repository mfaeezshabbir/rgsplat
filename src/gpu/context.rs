/// gpu/context.rs — wgpu device/queue initialization and buffer utilities
///
/// Designed to:
/// - Reuse device/queue across frames (expensive to recreate)
/// - Pool GPU buffers to minimize allocation overhead
/// - Async-friendly: all submissions return futures

#[cfg(feature = "gpu")]
use wgpu::*;
#[cfg(feature = "gpu")]
use anyhow::Result;

#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub adapter_info: AdapterInfo,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Initialize GPU context. Prefers discrete GPU, falls back to integrated.
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;

        let info = adapter.get_info();
        tracing::info!(
            "GPU: {} ({:?}), backend: {:?}",
            info.name, info.device_type, info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("rgsplat_device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .await?;

        Ok(Self { device, queue, adapter_info: info })
    }

    /// Human-readable adapter name.
    pub fn name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Upload data to a GPU buffer (STORAGE | COPY_DST).
    pub fn upload_buffer<T: bytemuck::Pod>(&self, data: &[T], label: &str) -> Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        })
    }

    /// Create an empty GPU buffer for compute output.
    pub fn output_buffer(&self, size_bytes: u64, label: &str) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Readback buffer: GPU → CPU transfer staging
    pub fn readback_buffer(&self, size_bytes: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some("readback"),
            size: size_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Submit a command encoder and wait for completion.
    pub async fn submit_and_wait(&self, encoder: CommandEncoder) {
        let idx = self.queue.submit([encoder.finish()]);
        self.device.poll(Maintain::WaitForSubmissionIndex(idx));
    }

    /// Read back a buffer to CPU memory.
    pub async fn readback<T: bytemuck::Pod + Clone>(&self, buf: &Buffer, count: usize) -> Vec<T> {
        let slice = buf.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        slice.map_async(MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(Maintain::Wait);
        rx.await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        buf.unmap();
        result
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub async fn new() -> anyhow::Result<Self> {
        anyhow::bail!("GPU feature not enabled. Build with --features gpu")
    }

    pub fn name(&self) -> &str {
        "none"
    }
}