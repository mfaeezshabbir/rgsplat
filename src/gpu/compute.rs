//! gpu/compute.rs — GPU-accelerated projection via wgpu compute.
//!
//! [`GpuRenderer`] runs the `project` compute pass in `splat.wgsl` (one thread
//! per Gaussian) to transform Gaussians into screen-space `SplatInfo2D`, reads
//! them back, then alpha-composites on the CPU. This accelerates the expensive
//! per-Gaussian projection while keeping the compositor simple; it is intended
//! for **inference/preview** (the GPU pass uses degree-0 SH color only). Training
//! uses the differentiable CPU path in [`crate::cpu::backward`].
//!
//! Buffer layouts here mirror the WGSL `std430`/`uniform` layouts exactly
//! (note the explicit padding), so `bytemuck` casts upload correctly.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::*;

use crate::math::{Camera, Gaussian3D};

use super::GpuContext;
use super::sort::depth_order;

/// GPU-side Gaussian, matching the WGSL `Gaussian3D` `std430` layout (240 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGaussian {
    position: [f32; 3],
    opacity_logit: f32,
    rotation: [f32; 4],
    log_scale: [f32; 3],
    _pad: f32,
    sh: [f32; 48],
}

impl From<&Gaussian3D> for GpuGaussian {
    fn from(g: &Gaussian3D) -> Self {
        Self {
            position: g.position,
            opacity_logit: g.opacity_logit,
            rotation: g.rotation,
            log_scale: g.log_scale,
            _pad: 0.0,
            sh: g.sh_coeffs,
        }
    }
}

/// Camera uniforms, matching the WGSL `CameraUniforms` layout (176 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    _pad: [f32; 3],
}

/// Projected splat, matching the WGSL `SplatInfo2D` layout (40 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct SplatInfo2D {
    pub sx: f32,
    pub sy: f32,
    pub conic_a: f32,
    pub conic_b: f32,
    pub conic_c: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub alpha: f32,
    pub depth: f32,
}

/// GPU renderer holding a reusable device, projection pipeline and layout.
pub struct GpuRenderer {
    ctx: GpuContext,
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuRenderer {
    pub async fn new() -> Result<Self> {
        let ctx = GpuContext::new().await?;
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("splat"),
            source: ShaderSource::Wgsl(include_str!("splat.wgsl").into()),
        });

        let storage = |read_only: bool| BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = ctx.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("project_bgl"),
            entries: &[
                bgl_entry(0, storage(true)),  // gaussians
                bgl_entry(1, storage(false)), // splats_out
                bgl_entry(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // camera
                bgl_entry(3, storage(false)), // depth_keys
            ],
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("project_pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("project"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "project",
            compilation_options: PipelineCompilationOptions::default(),
        });

        Ok(Self { ctx, pipeline, layout })
    }

    /// Adapter name (e.g. for logging).
    pub fn adapter_name(&self) -> &str {
        self.ctx.name()
    }

    /// Project Gaussians on the GPU and return screen-space splats.
    pub async fn project(&self, gaussians: &[Gaussian3D], camera: &Camera) -> Result<Vec<SplatInfo2D>> {
        let n = gaussians.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let gpu_g: Vec<GpuGaussian> = gaussians.iter().map(GpuGaussian::from).collect();

        let view = camera.view_matrix();
        let proj = camera.projection_matrix(0.01, 1000.0);
        let cam_pos = camera.position();
        let uniforms = CameraUniforms {
            view: mat_to_cols(&view),
            proj: mat_to_cols(&proj),
            cam_pos,
            width: camera.width,
            height: camera.height,
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
            _pad: [0.0; 3],
        };

        let dev = &self.ctx.device;
        let gaussian_buf = dev.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("gaussians"),
            contents: bytemuck::cast_slice(&gpu_g),
            usage: BufferUsages::STORAGE,
        });
        let splat_size = (n * std::mem::size_of::<SplatInfo2D>()) as u64;
        let splats_buf = self.ctx.output_buffer(splat_size, "splats_out");
        let keys_buf = self.ctx.output_buffer((n * 4) as u64, "depth_keys");
        let camera_buf = dev.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = dev.create_bind_group(&BindGroupDescriptor {
            label: Some("project_bg"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: gaussian_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: splats_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: camera_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: keys_buf.as_entire_binding() },
            ],
        });

        let mut encoder = dev.create_command_encoder(&CommandEncoderDescriptor { label: Some("project") });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("project_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = ((n + 255) / 256) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // Copy results into a mappable buffer and read back.
        let readback = self.ctx.readback_buffer(splat_size);
        encoder.copy_buffer_to_buffer(&splats_buf, 0, &readback, 0, splat_size);
        self.ctx.submit_and_wait(encoder).await;
        let splats: Vec<SplatInfo2D> = self.ctx.readback(&readback, n).await;
        Ok(splats)
    }

    /// Full GPU-projected, CPU-composited render. Returns linear RGB (`w·h·3`).
    pub async fn render(
        &self,
        gaussians: &[Gaussian3D],
        camera: &Camera,
        bg: [f32; 3],
    ) -> Result<Vec<f32>> {
        let splats = self.project(gaussians, camera).await?;
        Ok(composite(&splats, camera.width as usize, camera.height as usize, bg))
    }
}

fn bgl_entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// nalgebra is column-major in memory; emit columns so the WGSL `mat4x4` (which
/// indexes `m[col]`) matches.
fn mat_to_cols(m: &nalgebra::Matrix4<f32>) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = m[(r, c)];
        }
    }
    out
}

/// CPU alpha compositor over already-projected splats (front-to-back).
fn composite(splats: &[SplatInfo2D], w: usize, h: usize, bg: [f32; 3]) -> Vec<f32> {
    let order = depth_order(splats);
    let mut color = vec![0.0f32; w * h * 3];
    let mut trans = vec![1.0f32; w * h];

    for &i in &order {
        let s = splats[i];
        // Screen-space radius from the conic (Σ⁻¹): r = 3·sqrt(1/λ_min(conic)).
        let (a, b, c) = (s.conic_a, s.conic_b, s.conic_c);
        let mid = 0.5 * (a + c);
        let disc = (0.25 * (a - c) * (a - c) + b * b).sqrt();
        let lambda_min = (mid - disc).max(1e-6);
        let rad = 3.0 * (1.0 / lambda_min).sqrt();

        let x0 = ((s.sx - rad).floor().max(0.0)) as usize;
        let y0 = ((s.sy - rad).floor().max(0.0)) as usize;
        let x1 = ((s.sx + rad).ceil() as i64).clamp(0, w as i64) as usize;
        let y1 = ((s.sy + rad).ceil() as i64).clamp(0, h as i64) as usize;

        for py in y0..y1 {
            for px in x0..x1 {
                let pix = py * w + px;
                let t = trans[pix];
                if t < 1e-4 {
                    continue;
                }
                let dx = px as f32 + 0.5 - s.sx;
                let dy = py as f32 + 0.5 - s.sy;
                let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
                if power > 0.0 {
                    continue;
                }
                let alpha = (s.alpha * power.exp()).min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let weight = alpha * t;
                color[pix * 3] += s.color_r * weight;
                color[pix * 3 + 1] += s.color_g * weight;
                color[pix * 3 + 2] += s.color_b * weight;
                trans[pix] = t * (1.0 - alpha);
            }
        }
    }

    // Composite the remaining transmittance over the background.
    for pix in 0..w * h {
        let t = trans[pix];
        color[pix * 3] += t * bg[0];
        color[pix * 3 + 1] += t * bg[1];
        color[pix * 3 + 2] += t * bg[2];
    }
    color
}
