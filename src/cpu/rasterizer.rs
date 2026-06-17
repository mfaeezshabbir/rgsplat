/// cpu/rasterizer.rs — Tile-based Gaussian rasterizer (CPU, Rayon parallel)
///
/// Algorithm:
///   1. Project each Gaussian to screen space
///   2. Sort by depth (back-to-front)
///   3. Divide screen into 16×16 tiles
///   4. For each tile, alpha-composite Gaussians front-to-back:
///      C_out = C_out + α·(1 - A_acc)·c_i
///      A_acc = A_acc + α·(1 - A_acc)
///      Stop when A_acc > 0.9999 (early termination)
///
/// Parallelism: Rayon par_iter over tiles — each tile independent.

use rayon::prelude::*;
use crate::math::{Gaussian3D, Camera, projection, sh};

const TILE_SIZE: usize = 16;
const ALPHA_THRESHOLD: f32 = 1.0 / 255.0;  // cull sub-pixel Gaussians
const TRANSMITTANCE_STOP: f32 = 0.0001;    // early ray termination

/// Output framebuffer: RGBA f32 (linear, pre-multiplied internally)
pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    /// Row-major: pixel (x, y) at index (y * width + x) * 4
    pub data: Vec<f32>,   // RGBA
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width, height,
            data: vec![0.0; width * height * 4],
        }
    }

    /// Convert to u8 RGBA (gamma-corrected, sRGB)
    pub fn to_u8_srgb(&self) -> Vec<u8> {
        self.data.iter().enumerate().map(|(i, &v)| {
            if i % 4 == 3 {
                // alpha: linear
                (v.clamp(0.0, 1.0) * 255.0) as u8
            } else {
                // RGB: linear → sRGB  (γ ≈ 2.2)
                (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0) as u8
            }
        }).collect()
    }
}

/// Projected Gaussian: everything needed for 2D rasterization
#[derive(Clone)]
struct SplatInfo {
    /// Screen-space center (pixels)
    sx: f32, sy: f32,
    /// Inverse 2×2 covariance for Gaussian eval: [a, b, c] where Σ⁻¹ = [[a,b],[b,c]]
    conic: [f32; 3],
    /// View-dependent color [R, G, B]
    color: [f32; 3],
    /// Opacity α
    alpha: f32,
    /// Depth (for sorting)
    depth: f32,
    /// Screen-space AABB (tile range)
    tile_x0: i32, tile_y0: i32, tile_x1: i32, tile_y1: i32,
}

/// Rasterize a slice of Gaussians into a framebuffer.
///
/// This is the main CPU entry point. Non-blocking: call from a Tokio
/// spawn_blocking context to avoid stalling the async executor.
pub fn rasterize(
    gaussians: &[Gaussian3D],
    camera: &Camera,
    bg_color: [f32; 3],
) -> Framebuffer {
    let (w, h) = (camera.width as usize, camera.height as usize);
    let view = camera.view_matrix();
    let focal = camera.focal();

    // Camera position in world space (for SH view direction)
    let cam_pos = camera.position();

    // ── Step 1: Project all Gaussians ─────────────────────────────────────
    let n_tiles_x = (w + TILE_SIZE - 1) / TILE_SIZE;
    let n_tiles_y = (h + TILE_SIZE - 1) / TILE_SIZE;

    let mut splats: Vec<SplatInfo> = gaussians
        .par_iter()
        .filter_map(|g| project_gaussian(g, camera, &view, focal, cam_pos, w, h))
        .collect();

    // ── Step 2: Sort front-to-back (nearest first) ───────────────────────
    // Front-to-back compositing with a running transmittance requires the
    // nearest splat first (ascending depth).
    splats.sort_unstable_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

    // ── Step 3: Tile-parallel rasterization ───────────────────────────────
    // Build per-tile splat lists (assign each splat to overlapping tiles)
    let n_tiles = n_tiles_x * n_tiles_y;
    let mut tile_lists: Vec<Vec<usize>> = vec![Vec::new(); n_tiles];

    for (idx, s) in splats.iter().enumerate() {
        let tx0 = (s.tile_x0).max(0) as usize;
        let ty0 = (s.tile_y0).max(0) as usize;
        let tx1 = (s.tile_x1 as usize).min(n_tiles_x);
        let ty1 = (s.tile_y1 as usize).min(n_tiles_y);
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                tile_lists[ty * n_tiles_x + tx].push(idx);
            }
        }
    }

    // Pixel buffer (flat, RGBA)
    let pixels = vec![0.0f32; w * h * 4];

    // Rayon: each tile processed independently — zero lock contention
    tile_lists
        .par_iter()
        .enumerate()
        .for_each(|(tile_idx, splat_indices)| {
            if splat_indices.is_empty() { return; }

            let tx = tile_idx % n_tiles_x;
            let ty = tile_idx / n_tiles_x;
            let px0 = tx * TILE_SIZE;
            let py0 = ty * TILE_SIZE;
            let px1 = (px0 + TILE_SIZE).min(w);
            let py1 = (py0 + TILE_SIZE).min(h);

            // Temporary tile pixel buffer
            let mut tile_buf = vec![[bg_color[0], bg_color[1], bg_color[2], 1.0f32]; TILE_SIZE * TILE_SIZE];
            let mut transmittance = vec![1.0f32; TILE_SIZE * TILE_SIZE];

            // Alpha-composite splats front-to-back within tile
            for &si in splat_indices.iter() {
                let s = &splats[si];

                for py in py0..py1 {
                    for px in px0..px1 {
                        let local = (py - py0) * TILE_SIZE + (px - px0);
                        let t = transmittance[local];
                        if t < TRANSMITTANCE_STOP { continue; }

                        // Evaluate Gaussian at pixel center
                        let dx = px as f32 + 0.5 - s.sx;
                        let dy = py as f32 + 0.5 - s.sy;

                        // Power = -½ [dx dy] Σ⁻¹ [dx dy]ᵀ
                        // Σ⁻¹ = [[a, b], [b, c]]  (conic representation)
                        let [a, b, c] = s.conic;
                        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);

                        if power > 0.0 { continue; }  // outside Gaussian

                        let alpha = (s.alpha * power.exp()).min(0.99);
                        if alpha < ALPHA_THRESHOLD { continue; }

                        // Front-to-back alpha compositing
                        let weight = alpha * t;
                        let p = &mut tile_buf[local];
                        p[0] += s.color[0] * weight;
                        p[1] += s.color[1] * weight;
                        p[2] += s.color[2] * weight;
                        p[3] -= weight;  // will finalize to alpha
                        transmittance[local] *= 1.0 - alpha;
                    }
                }
            }

            // Write tile into global pixel buffer (no overlap between tiles → safe)
            // Using raw pointer arithmetic for zero-overhead tile copy
            let pixels_ptr = pixels.as_ptr() as *mut f32;
            for py in py0..py1 {
                for px in px0..px1 {
                    let local = (py - py0) * TILE_SIZE + (px - px0);
                    let global = (py * w + px) * 4;
                    let p = tile_buf[local];
                    // SAFETY: each pixel written by exactly one tile — no aliasing
                    unsafe {
                        *pixels_ptr.add(global)     = p[0];
                        *pixels_ptr.add(global + 1) = p[1];
                        *pixels_ptr.add(global + 2) = p[2];
                        *pixels_ptr.add(global + 3) = 1.0 - transmittance[local];
                    }
                }
            }
        });

    Framebuffer { width: w, height: h, data: pixels }
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection helper
// ─────────────────────────────────────────────────────────────────────────────

fn project_gaussian(
    g: &Gaussian3D,
    camera: &Camera,
    view: &nalgebra::Matrix4<f32>,
    focal: (f32, f32),
    cam_pos: [f32; 3],
    w: usize, h: usize,
) -> Option<SplatInfo> {
    use nalgebra::Vector4;

    let alpha = g.opacity();
    if alpha < ALPHA_THRESHOLD { return None; }

    // Transform to camera space
    let p = Vector4::new(g.position[0], g.position[1], g.position[2], 1.0);
    let p_cam = view * p;
    let tz = p_cam.z;
    if tz <= 0.001 { return None; }  // behind camera

    let (fx, fy) = focal;
    let (cx, cy) = (camera.cx, camera.cy);

    // Project to screen
    let sx = fx * p_cam.x / tz + cx;
    let sy = fy * p_cam.y / tz + cy;

    // Early cull: center outside screen (with margin for large Gaussians)
    let margin = 128.0;
    if sx < -margin || sx > w as f32 + margin || sy < -margin || sy > h as f32 + margin {
        return None;
    }

    // Project 3D covariance → 2D
    let sigma_2d = projection::project_covariance(
        &g.covariance_3d(), &g.position, view, focal,
    );

    // Extract 2×2 sub-block and invert to get conic = Σ⁻¹
    let a = sigma_2d[(0, 0)];
    let b = sigma_2d[(0, 1)];  // = sigma_2d[(1, 0)] by symmetry
    let c = sigma_2d[(1, 1)];
    let det = a * c - b * b;
    if det.abs() < 1e-8 { return None; }  // degenerate
    let inv_det = 1.0 / det;
    let conic = [c * inv_det, -b * inv_det, a * inv_det];

    // View-dependent color via SH
    let dir = sh::view_dir(g.position, cam_pos);
    let color = sh::eval_sh_deg3(&g.sh_coeffs, dir);

    // Tile-space AABB (n_sigma = 3 covers ~99.7% mass)
    let aabb = projection::gaussian_aabb_2d(
        (sx, sy), &sigma_2d, 3.0, camera.width, camera.height,
    )?;
    let (ax0, ay0, ax1, ay1) = aabb;

    Some(SplatInfo {
        sx, sy,
        conic,
        color,
        alpha,
        depth: tz,
        tile_x0: ax0 / TILE_SIZE as i32,
        tile_y0: ay0 / TILE_SIZE as i32,
        tile_x1: (ax1 + TILE_SIZE as i32 - 1) / TILE_SIZE as i32,
        tile_y1: (ay1 + TILE_SIZE as i32 - 1) / TILE_SIZE as i32,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Color utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Linear to sRGB gamma correction (IEC 61966-2-1)
#[inline(always)]
fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.003130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_srgb() {
        assert!((linear_to_srgb(0.0) - 0.0).abs() < 1e-6);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        assert!(linear_to_srgb(0.5) > 0.5);  // sRGB brighter than linear
    }
}