//! cpu/backward.rs — differentiable CPU rasterizer + photometric loss.
//!
//! Implements the analytic backward pass of tile-based Gaussian splatting:
//! a forward render that caches the per-pixel transmittance and contributor
//! count, then a reverse pass that turns `∂L/∂pixel` into per-Gaussian gradients
//! on SH color, opacity, screen-space mean (→ position) and the 2D conic
//! (→ covariance → scale & rotation).
//!
//! The loss is the 3DGS objective `L = (1−λ)·L1 + λ·(1 − SSIM)` with `λ = 0.2`,
//! and its exact pixel-space gradient.
//!
//! Gradients are validated against finite differences in the tests below and in
//! `tests/gradient_check.rs`.

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use rayon::prelude::*;

use crate::io::TargetImage;
use crate::math::{Camera, Gaussian3D, covariance, projection, sh};

const TILE_SIZE: usize = 16;
const ALPHA_THRESHOLD: f32 = 1.0 / 255.0;
const TRANSMITTANCE_STOP: f32 = 1e-4;
const MAX_ALPHA: f32 = 0.99;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Per-Gaussian gradient. Field order matches the optimizer's flat layout:
/// position(3), rotation(4), log_scale(3), opacity(1), sh(48).
#[derive(Clone, Debug)]
pub struct GaussianGrad {
    pub d_position: [f32; 3],
    pub d_rotation: [f32; 4],
    pub d_log_scale: [f32; 3],
    pub d_opacity_logit: f32,
    pub d_sh: [f32; 48],
    /// Screen-space mean gradient magnitude (densification signal).
    pub screen_grad_norm: f32,
}

impl Default for GaussianGrad {
    fn default() -> Self {
        Self {
            d_position: [0.0; 3],
            d_rotation: [0.0; 4],
            d_log_scale: [0.0; 3],
            d_opacity_logit: 0.0,
            d_sh: [0.0; 48],
            screen_grad_norm: 0.0,
        }
    }
}

/// Loss weighting.
#[derive(Clone, Copy, Debug)]
pub struct LossConfig {
    /// Weight of the D-SSIM term (`λ` in the 3DGS objective).
    pub lambda_dssim: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self { lambda_dssim: 0.2 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Projected splat (forward cache)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Splat {
    gi: usize, // source Gaussian index
    sx: f32,
    sy: f32,
    conic: [f32; 3], // Σ⁻¹ = [[a,b],[b,c]]
    cov2d: [f32; 3], // (A, B, C) low-passed 2D covariance
    color: [f32; 3], // clamped, as composited
    raw_color: [f32; 3],
    opacity: f32, // sigmoid(logit)
    depth: f32,
    basis: [f32; 16],
    pcam: [f32; 3],
    tile_x0: i32,
    tile_y0: i32,
    tile_x1: i32,
    tile_y1: i32,
}

#[derive(Clone, Default)]
struct SplatGradAccum {
    color: [f32; 3],
    opacity: f32,
    conic: [f32; 3],
    sx: f32,
    sy: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level entry: render + loss + gradients
// ─────────────────────────────────────────────────────────────────────────────

/// Render `gaussians` from `camera`, compare against `target`, and return the
/// scalar loss together with per-Gaussian gradients.
pub fn render_and_backward(
    gaussians: &[Gaussian3D],
    camera: &Camera,
    target: &TargetImage,
    bg: [f32; 3],
    cfg: &LossConfig,
) -> (f32, Vec<GaussianGrad>) {
    let w = camera.width as usize;
    let h = camera.height as usize;
    let view = camera.view_matrix();
    let focal = camera.focal();
    let cam_pos = camera.position();

    // ── Forward ─────────────────────────────────────────────────────────────────
    let (splats, tile_lists, n_tiles_x) = project_and_bin(gaussians, camera, &view, focal, cam_pos, w, h);

    // Per-pixel forward state.
    let mut rendered = vec![0.0f32; w * h * 3]; // RGB over bg
    let mut final_t = vec![1.0f32; w * h];
    let mut last_contrib = vec![-1i32; w * h];

    forward_pass(
        &splats, &tile_lists, n_tiles_x, w, h, bg, &mut rendered, &mut final_t, &mut last_contrib,
    );

    // ── Loss + dL/dpixel ─────────────────────────────────────────────────────────
    let (loss, dl_dpix) = photometric_loss_grad(&rendered, &target.data, w, h, cfg.lambda_dssim);

    // ── Backward ──────────────────────────────────────────────────────────────────
    let splat_grads = backward_pass(
        &splats, &tile_lists, n_tiles_x, w, h, bg, &final_t, &last_contrib, &dl_dpix,
    );

    // Geometry backprop: per splat → per Gaussian.
    let mut grads = vec![GaussianGrad::default(); gaussians.len()];
    let view3 = view.fixed_view::<3, 3>(0, 0).into_owned();
    for (si, splat) in splats.iter().enumerate() {
        let acc = &splat_grads[si];
        grads[splat.gi] = geometry_backprop(splat, acc, gaussians, &view3, focal, cam_pos);
    }

    (loss, grads)
}

/// Forward-only render to an RGB buffer (used by inference / tests).
pub fn render_color(gaussians: &[Gaussian3D], camera: &Camera, bg: [f32; 3]) -> Vec<f32> {
    let w = camera.width as usize;
    let h = camera.height as usize;
    let view = camera.view_matrix();
    let focal = camera.focal();
    let cam_pos = camera.position();
    let (splats, tile_lists, n_tiles_x) = project_and_bin(gaussians, camera, &view, focal, cam_pos, w, h);
    let mut rendered = vec![0.0f32; w * h * 3];
    let mut final_t = vec![1.0f32; w * h];
    let mut last_contrib = vec![-1i32; w * h];
    forward_pass(
        &splats, &tile_lists, n_tiles_x, w, h, bg, &mut rendered, &mut final_t, &mut last_contrib,
    );
    rendered
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection + tile binning
// ─────────────────────────────────────────────────────────────────────────────

fn project_and_bin(
    gaussians: &[Gaussian3D],
    camera: &Camera,
    view: &Matrix4<f32>,
    focal: (f32, f32),
    cam_pos: [f32; 3],
    w: usize,
    h: usize,
) -> (Vec<Splat>, Vec<Vec<usize>>, usize) {
    let n_tiles_x = (w + TILE_SIZE - 1) / TILE_SIZE;
    let n_tiles_y = (h + TILE_SIZE - 1) / TILE_SIZE;

    let mut splats: Vec<Splat> = gaussians
        .par_iter()
        .enumerate()
        .filter_map(|(gi, g)| project(g, gi, camera, view, focal, cam_pos, w, h))
        .collect();

    // Front-to-back ordering (nearest first).
    splats.sort_unstable_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

    let n_tiles = n_tiles_x * n_tiles_y;
    let mut tile_lists: Vec<Vec<usize>> = vec![Vec::new(); n_tiles];
    for (idx, s) in splats.iter().enumerate() {
        let tx0 = s.tile_x0.max(0) as usize;
        let ty0 = s.tile_y0.max(0) as usize;
        let tx1 = (s.tile_x1 as usize).min(n_tiles_x);
        let ty1 = (s.tile_y1 as usize).min(n_tiles_y);
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                tile_lists[ty * n_tiles_x + tx].push(idx);
            }
        }
    }

    (splats, tile_lists, n_tiles_x)
}

fn project(
    g: &Gaussian3D,
    gi: usize,
    camera: &Camera,
    view: &Matrix4<f32>,
    focal: (f32, f32),
    cam_pos: [f32; 3],
    w: usize,
    h: usize,
) -> Option<Splat> {
    let opacity = g.opacity();
    if opacity < ALPHA_THRESHOLD {
        return None;
    }

    let p = Vector4::new(g.position[0], g.position[1], g.position[2], 1.0);
    let p_cam = view * p;
    let tz = p_cam.z;
    if tz <= 0.001 {
        return None;
    }

    let (fx, fy) = focal;
    let sx = fx * p_cam.x / tz + camera.cx;
    let sy = fy * p_cam.y / tz + camera.cy;

    let margin = 128.0;
    if sx < -margin || sx > w as f32 + margin || sy < -margin || sy > h as f32 + margin {
        return None;
    }

    let sigma_2d = projection::project_covariance(&g.covariance_3d(), &g.position, view, focal);
    let a = sigma_2d[(0, 0)];
    let b = sigma_2d[(0, 1)];
    let c = sigma_2d[(1, 1)];
    let det = a * c - b * b;
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    let conic = [c * inv_det, -b * inv_det, a * inv_det];

    let basis = sh::sh_basis(sh::view_dir(g.position, cam_pos));
    let raw_color = sh::eval_sh_raw(&g.sh_coeffs, &basis);
    let color = [
        raw_color[0].clamp(0.0, 1.0),
        raw_color[1].clamp(0.0, 1.0),
        raw_color[2].clamp(0.0, 1.0),
    ];

    let (ax0, ay0, ax1, ay1) =
        projection::gaussian_aabb_2d((sx, sy), &sigma_2d, 3.0, camera.width, camera.height)?;

    Some(Splat {
        gi,
        sx,
        sy,
        conic,
        cov2d: [a, b, c],
        color,
        raw_color,
        opacity,
        depth: tz,
        basis,
        pcam: [p_cam.x, p_cam.y, p_cam.z],
        tile_x0: ax0 / TILE_SIZE as i32,
        tile_y0: ay0 / TILE_SIZE as i32,
        tile_x1: (ax1 + TILE_SIZE as i32 - 1) / TILE_SIZE as i32,
        tile_y1: (ay1 + TILE_SIZE as i32 - 1) / TILE_SIZE as i32,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass (caches final_t + last_contrib for the backward pass)
// ─────────────────────────────────────────────────────────────────────────────

struct TileForward {
    px0: usize,
    py0: usize,
    px1: usize,
    py1: usize,
    rgb: Vec<[f32; 3]>,
    t: Vec<f32>,
    last: Vec<i32>,
}

fn forward_pass(
    splats: &[Splat],
    tile_lists: &[Vec<usize>],
    n_tiles_x: usize,
    w: usize,
    h: usize,
    bg: [f32; 3],
    rendered: &mut [f32],
    final_t: &mut [f32],
    last_contrib: &mut [i32],
) {
    // Each tile composites its own pixels in parallel into local buffers; we
    // then scatter the results into the (disjoint) image regions sequentially.
    let tiles: Vec<TileForward> = tile_lists
        .par_iter()
        .enumerate()
        .filter_map(|(tile_idx, splat_indices)| {
            if splat_indices.is_empty() {
                return None;
            }
            let tx = tile_idx % n_tiles_x;
            let ty = tile_idx / n_tiles_x;
            let px0 = tx * TILE_SIZE;
            let py0 = ty * TILE_SIZE;
            let px1 = (px0 + TILE_SIZE).min(w);
            let py1 = (py0 + TILE_SIZE).min(h);
            let tw = px1 - px0;
            let th = py1 - py0;

            let mut rgb = vec![[0.0f32; 3]; tw * th];
            let mut t_buf = vec![1.0f32; tw * th];
            let mut last_buf = vec![-1i32; tw * th];

            for py in py0..py1 {
                for px in px0..px1 {
                    let mut t = 1.0f32;
                    let mut col = [0.0f32; 3];
                    let mut last = -1i32;

                    for (local_i, &si) in splat_indices.iter().enumerate() {
                        let s = &splats[si];
                        let dx = px as f32 + 0.5 - s.sx;
                        let dy = py as f32 + 0.5 - s.sy;
                        let [a, b, c] = s.conic;
                        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
                        if power > 0.0 {
                            continue;
                        }
                        let alpha = (s.opacity * power.exp()).min(MAX_ALPHA);
                        if alpha < ALPHA_THRESHOLD {
                            continue;
                        }
                        let weight = alpha * t;
                        col[0] += s.color[0] * weight;
                        col[1] += s.color[1] * weight;
                        col[2] += s.color[2] * weight;
                        t *= 1.0 - alpha;
                        last = local_i as i32;
                        if t < TRANSMITTANCE_STOP {
                            break;
                        }
                    }

                    let local = (py - py0) * tw + (px - px0);
                    rgb[local] = [col[0] + t * bg[0], col[1] + t * bg[1], col[2] + t * bg[2]];
                    t_buf[local] = t;
                    last_buf[local] = last;
                }
            }

            Some(TileForward { px0, py0, px1, py1, rgb, t: t_buf, last: last_buf })
        })
        .collect();

    for tile in tiles {
        let tw = tile.px1 - tile.px0;
        for py in tile.py0..tile.py1 {
            for px in tile.px0..tile.px1 {
                let local = (py - tile.py0) * tw + (px - tile.px0);
                let pix = py * w + px;
                let c = tile.rgb[local];
                rendered[pix * 3] = c[0];
                rendered[pix * 3 + 1] = c[1];
                rendered[pix * 3 + 2] = c[2];
                final_t[pix] = tile.t[local];
                last_contrib[pix] = tile.last[local];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward pass over the composite (per-pixel reverse traversal)
// ─────────────────────────────────────────────────────────────────────────────

fn backward_pass(
    splats: &[Splat],
    tile_lists: &[Vec<usize>],
    n_tiles_x: usize,
    w: usize,
    h: usize,
    bg: [f32; 3],
    final_t: &[f32],
    last_contrib: &[i32],
    dl_dpix: &[f32],
) -> Vec<SplatGradAccum> {
    // Each tile produces sparse (global splat idx, accum) pairs; merged after.
    let per_tile: Vec<Vec<(usize, SplatGradAccum)>> = tile_lists
        .par_iter()
        .enumerate()
        .map(|(tile_idx, splat_indices)| {
            if splat_indices.is_empty() {
                return Vec::new();
            }
            let tx = tile_idx % n_tiles_x;
            let ty = tile_idx / n_tiles_x;
            let px0 = tx * TILE_SIZE;
            let py0 = ty * TILE_SIZE;
            let px1 = (px0 + TILE_SIZE).min(w);
            let py1 = (py0 + TILE_SIZE).min(h);

            // Local accumulators aligned to this tile's splat list.
            let mut local = vec![SplatGradAccum::default(); splat_indices.len()];

            for py in py0..py1 {
                for px in px0..px1 {
                    let pix = py * w + px;
                    let lc = last_contrib[pix];
                    if lc < 0 {
                        continue;
                    }
                    let dl = [dl_dpix[pix * 3], dl_dpix[pix * 3 + 1], dl_dpix[pix * 3 + 2]];

                    let mut t = final_t[pix];
                    // `behind` = color contributed by everything strictly behind the
                    // current splat (including the background leak T·bg).
                    let mut behind = [t * bg[0], t * bg[1], t * bg[2]];

                    for local_i in (0..=lc as usize).rev() {
                        let si = splat_indices[local_i];
                        let s = &splats[si];
                        let dx = px as f32 + 0.5 - s.sx;
                        let dy = py as f32 + 0.5 - s.sy;
                        let [a, b, c] = s.conic;
                        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
                        if power > 0.0 {
                            continue;
                        }
                        let g_exp = power.exp();
                        let alpha_raw = s.opacity * g_exp;
                        let alpha = alpha_raw.min(MAX_ALPHA);
                        if alpha < ALPHA_THRESHOLD {
                            continue;
                        }

                        // Reconstruct transmittance in front of this splat.
                        t /= 1.0 - alpha;
                        let weight = alpha * t;

                        let g = &mut local[local_i];

                        // dL/dcolor
                        g.color[0] += dl[0] * weight;
                        g.color[1] += dl[1] * weight;
                        g.color[2] += dl[2] * weight;

                        // dL/dalpha = Σ_ch dL/dC · (T·c − behind/(1−α))
                        let inv_1ma = 1.0 / (1.0 - alpha);
                        let mut dl_dalpha = 0.0f32;
                        for ch in 0..3 {
                            dl_dalpha += dl[ch] * (t * s.color[ch] - behind[ch] * inv_1ma);
                        }

                        // Through the alpha cap and α = opacity · exp(power).
                        let dalpha_draw = if alpha_raw > MAX_ALPHA { 0.0 } else { 1.0 };
                        let dl_draw = dl_dalpha * dalpha_draw;
                        g.opacity += dl_draw * g_exp;

                        let dl_dpower = dl_draw * s.opacity * g_exp;
                        g.conic[0] += dl_dpower * (-0.5 * dx * dx);
                        g.conic[1] += dl_dpower * (-dx * dy);
                        g.conic[2] += dl_dpower * (-0.5 * dy * dy);
                        g.sx += dl_dpower * (a * dx + b * dy);
                        g.sy += dl_dpower * (b * dx + c * dy);

                        // Advance `behind` to include this splat for the next iter.
                        behind[0] += weight * s.color[0];
                        behind[1] += weight * s.color[1];
                        behind[2] += weight * s.color[2];
                    }
                }
            }

            local
                .into_iter()
                .enumerate()
                .map(|(li, acc)| (splat_indices[li], acc))
                .collect()
        })
        .collect();

    // Merge sparse per-tile accumulators into one entry per splat.
    let mut merged = vec![SplatGradAccum::default(); splats.len()];
    for tile in &per_tile {
        for (idx, acc) in tile {
            let m = &mut merged[*idx];
            for ch in 0..3 {
                m.color[ch] += acc.color[ch];
                m.conic[ch] += acc.conic[ch];
            }
            m.opacity += acc.opacity;
            m.sx += acc.sx;
            m.sy += acc.sy;
        }
    }
    merged
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry backprop: per-splat accumulators → per-Gaussian gradients
// ─────────────────────────────────────────────────────────────────────────────

fn geometry_backprop(
    s: &Splat,
    acc: &SplatGradAccum,
    gaussians: &[Gaussian3D],
    view3: &Matrix3<f32>,
    focal: (f32, f32),
    cam_pos: [f32; 3],
) -> GaussianGrad {
    let g = &gaussians[s.gi];
    let mut out = GaussianGrad::default();

    // ── Opacity: α = σ(logit) ⇒ dL/dlogit = dL/dα · σ(1−σ) ──────────────────────
    let op = s.opacity;
    out.d_opacity_logit = acc.opacity * op * (1.0 - op);

    // ── SH color (linear in the coefficients, masked by the [0,1] clamp) ─────────
    // `masked[ch]` is dL/draw_color for the channel (0 where the color saturates).
    let masked = [
        if s.raw_color[0] > 0.0 && s.raw_color[0] < 1.0 { acc.color[0] } else { 0.0 },
        if s.raw_color[1] > 0.0 && s.raw_color[1] < 1.0 { acc.color[1] } else { 0.0 },
        if s.raw_color[2] > 0.0 && s.raw_color[2] < 1.0 { acc.color[2] } else { 0.0 },
    ];
    for ch in 0..3 {
        for k in 0..16 {
            out.d_sh[ch * 16 + k] = masked[ch] * s.basis[k];
        }
    }

    // ── Conic → 2D covariance (A, B, C). conic = Σ2D⁻¹. ─────────────────────────
    let (ga, gb, gc) = (acc.conic[0], acc.conic[1], acc.conic[2]);
    let [a2, b2, c2] = s.cov2d; // covariance entries (post low-pass)
    let det = a2 * c2 - b2 * b2;
    let det2 = det * det;
    // dL/d{A,B,C} from dL/d{conic_a, conic_b, conic_c}.
    let dl_da = (-ga * c2 * c2 + gb * b2 * c2 - gc * b2 * b2) / det2;
    let dl_db = (2.0 * ga * b2 * c2 - gb * (det + 2.0 * b2 * b2) + 2.0 * gc * a2 * b2) / det2;
    let dl_dc = (-ga * b2 * b2 + gb * a2 * b2 - gc * a2 * a2) / det2;

    // ── 2D covariance → 3D covariance via Σ2D = T Σ3D Tᵀ, T = J W ───────────────
    let (fx, fy) = focal;
    let (tx, ty, tz) = (s.pcam[0], s.pcam[1], s.pcam[2]);
    let j = Matrix3::new(
        fx / tz, 0.0, -fx * tx / (tz * tz),
        0.0, fy / tz, -fy * ty / (tz * tz),
        0.0, 0.0, 0.0,
    );
    let t_mat = j * view3;
    let sigma3d = g.covariance_3d();

    // dL/dΣ3D = Tᵀ G T, with the symmetric off-diagonal carrying 0.5·dL/dB.
    let gcov = Matrix3::new(
        dl_da, 0.5 * dl_db, 0.0,
        0.5 * dl_db, dl_dc, 0.0,
        0.0, 0.0, 0.0,
    );
    let dl_dsigma3d = t_mat.transpose() * gcov * t_mat;
    let (d_log_scale, d_rot) = covariance::backprop_cov3d(g.scale(), g.rotation, &dl_dsigma3d);
    out.d_log_scale = d_log_scale;
    out.d_rotation = d_rot;

    // ── Screen mean (sx, sy) → camera-space position ────────────────────────────
    let mut dl_dpcam = Vector3::new(
        acc.sx * fx / tz,
        acc.sy * fy / tz,
        acc.sx * (-fx * tx / (tz * tz)) + acc.sy * (-fy * ty / (tz * tz)),
    );

    // ── Conic's dependence on the mean (J depends on pcam) → extra position grad ─
    // dΣ2D/dpcam_k = dT_k Σ3D Tᵀ + T Σ3D dT_kᵀ, contracted with (dL/dA,B,C).
    let dj = [
        // ∂J/∂tx
        Matrix3::new(0.0, 0.0, -fx / (tz * tz), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        // ∂J/∂ty
        Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -fy / (tz * tz), 0.0, 0.0, 0.0),
        // ∂J/∂tz
        Matrix3::new(
            -fx / (tz * tz), 0.0, 2.0 * fx * tx / (tz * tz * tz),
            0.0, -fy / (tz * tz), 2.0 * fy * ty / (tz * tz * tz),
            0.0, 0.0, 0.0,
        ),
    ];
    let sig_tt = sigma3d * t_mat.transpose(); // Σ3D Tᵀ
    for k in 0..3 {
        let dt = dj[k] * view3; // dT/dpcam_k
        let dsigma2d = dt * sig_tt + t_mat * sigma3d * dt.transpose();
        let contrib = dl_da * dsigma2d[(0, 0)] + dl_db * dsigma2d[(0, 1)] + dl_dc * dsigma2d[(1, 1)];
        dl_dpcam[k] += contrib;
    }

    // pcam = W·pos + t  ⇒  dL/dpos = Wᵀ · dL/dpcam.
    let mut dl_dpos = view3.transpose() * dl_dpcam;

    // ── View-dependent SH color depends on position via dir = (cam − pos)/‖·‖ ────
    // dL/dbasis_k = Σ_ch (dL/dcolor_ch)·coeff_k ; dL/ddir = Σ_k dL/dbasis_k·∂basis_k/∂dir.
    let u = Vector3::new(
        cam_pos[0] - g.position[0],
        cam_pos[1] - g.position[1],
        cam_pos[2] - g.position[2],
    );
    let len = u.norm();
    if len > 1e-8 {
        let dir = u / len;
        let dbasis = sh::sh_basis_grad([dir[0], dir[1], dir[2]]);
        let mut dl_ddir = Vector3::zeros();
        for k in 0..16 {
            let dl_dbasis_k = masked[0] * g.sh_coeffs[k]
                + masked[1] * g.sh_coeffs[16 + k]
                + masked[2] * g.sh_coeffs[32 + k];
            dl_ddir += dl_dbasis_k * Vector3::new(dbasis[k][0], dbasis[k][1], dbasis[k][2]);
        }
        // dir = u/‖u‖ with u = cam − pos. ∂dir/∂u = (I − dir·dirᵀ)/‖u‖, ∂u/∂pos = −I.
        // ⇒ dL/dpos += −(1/‖u‖)(I − dir·dirᵀ)·dL/ddir.
        let proj = dl_ddir - dir * dir.dot(&dl_ddir);
        dl_dpos += -(proj / len);
    }

    out.d_position = [dl_dpos[0], dl_dpos[1], dl_dpos[2]];

    out.screen_grad_norm = (acc.sx * acc.sx + acc.sy * acc.sy).sqrt();
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Photometric loss: L = (1−λ)·L1 + λ·(1 − SSIM), and dL/dpixel
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `(loss, dL/d_rendered)` where `rendered`/`dl` are interleaved RGB
/// (`w·h·3`) and `target` is interleaved RGBA (`w·h·4`).
pub fn photometric_loss_grad(
    rendered: &[f32],
    target_rgba: &[f32],
    w: usize,
    h: usize,
    lambda: f32,
) -> (f32, Vec<f32>) {
    let n = w * h;
    let m_total = (n * 3) as f32;

    // Split into planar channels for SSIM convolutions.
    let mut x = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let mut y = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    for p in 0..n {
        for ch in 0..3 {
            x[ch][p] = rendered[p * 3 + ch];
            y[ch][p] = target_rgba[p * 4 + ch];
        }
    }

    // ── L1 ────────────────────────────────────────────────────────────────────
    let mut l1 = 0.0f32;
    let mut dl = vec![0.0f32; n * 3];
    for p in 0..n {
        for ch in 0..3 {
            let d = rendered[p * 3 + ch] - target_rgba[p * 4 + ch];
            l1 += d.abs();
            dl[p * 3 + ch] = (1.0 - lambda) * d.signum() / m_total;
        }
    }
    let l1 = l1 / m_total;

    // ── SSIM (per channel, separable Gaussian window) ───────────────────────────
    let kernel = gaussian_kernel_1d(5, 1.5);
    let mut ssim_sum = 0.0f32;
    for ch in 0..3 {
        let (s_ch, grad_ch) = ssim_channel(&x[ch], &y[ch], w, h, &kernel);
        ssim_sum += s_ch;
        // dLoss/dx for the −λ·SSIM term (mean over all pixels and channels).
        for p in 0..n {
            dl[p * 3 + ch] += -lambda * grad_ch[p] / m_total;
        }
    }
    let ssim_mean = ssim_sum / m_total;

    let loss = (1.0 - lambda) * l1 + lambda * (1.0 - ssim_mean);
    (loss, dl)
}

/// SSIM for one channel. Returns `(Σ_p SSIM_p, ∂(Σ_p SSIM_p)/∂x)` (the sum, not
/// the mean — the caller divides by the total sample count).
fn ssim_channel(x: &[f32], y: &[f32], w: usize, h: usize, k: &[f32]) -> (f32, Vec<f32>) {
    const C1: f32 = 0.01 * 0.01;
    const C2: f32 = 0.03 * 0.03;
    let n = w * h;

    let mu_x = blur(x, w, h, k);
    let mu_y = blur(y, w, h, k);
    let xx: Vec<f32> = x.iter().map(|v| v * v).collect();
    let yy: Vec<f32> = y.iter().map(|v| v * v).collect();
    let xy: Vec<f32> = x.iter().zip(y).map(|(a, b)| a * b).collect();
    let mu_xx = blur(&xx, w, h, k);
    let mu_yy = blur(&yy, w, h, k);
    let mu_xy = blur(&xy, w, h, k);

    let mut ssim_sum = 0.0f32;
    let mut pmap = vec![0.0f32; n];
    let mut qmap = vec![0.0f32; n];
    let mut rmap = vec![0.0f32; n];
    let mut qmu = vec![0.0f32; n];
    let mut rmu = vec![0.0f32; n];

    for p in 0..n {
        let (mx, my) = (mu_x[p], mu_y[p]);
        let sx2 = mu_xx[p] - mx * mx;
        let sy2 = mu_yy[p] - my * my;
        let sxy = mu_xy[p] - mx * my;

        let a1 = 2.0 * mx * my + C1;
        let a2 = 2.0 * sxy + C2;
        let b1 = mx * mx + my * my + C1;
        let b2 = sx2 + sy2 + C2;
        let num = a1 * a2;
        let den = b1 * b2;
        ssim_sum += num / den;

        // ∂SSIM/∂μx, ∂SSIM/∂σxy, ∂SSIM/∂σx² grouped for the conv-backprop.
        let inv_den = 1.0 / den;
        let p_coef = inv_den * (2.0 * my * a2) - (num / (den * den)) * (2.0 * mx * b2);
        let q_coef = inv_den * (2.0 * a1);
        let r_coef = -(num / (den * den)) * b1;
        pmap[p] = p_coef;
        qmap[p] = q_coef;
        rmap[p] = r_coef;
        qmu[p] = q_coef * my;
        rmu[p] = r_coef * mx;
    }

    // dS/dx_p = blur(P) + y·blur(Q) − blur(Q·μy) + 2x·blur(R) − 2·blur(R·μx)
    let bp = blur(&pmap, w, h, k);
    let bq = blur(&qmap, w, h, k);
    let br = blur(&rmap, w, h, k);
    let bqmu = blur(&qmu, w, h, k);
    let brmu = blur(&rmu, w, h, k);

    let mut grad = vec![0.0f32; n];
    for p in 0..n {
        grad[p] = bp[p] + y[p] * bq[p] - bqmu[p] + 2.0 * x[p] * br[p] - 2.0 * brmu[p];
    }
    (ssim_sum, grad)
}

/// Build a normalized 1D Gaussian kernel with the given radius and sigma.
fn gaussian_kernel_1d(radius: usize, sigma: f32) -> Vec<f32> {
    let mut k = vec![0.0f32; 2 * radius + 1];
    let mut sum = 0.0;
    for i in 0..k.len() {
        let d = i as f32 - radius as f32;
        let v = (-0.5 * d * d / (sigma * sigma)).exp();
        k[i] = v;
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Separable Gaussian blur with **zero** padding (so the operator is symmetric /
/// self-adjoint, which the SSIM gradient relies on).
fn blur(src: &[f32], w: usize, h: usize, k: &[f32]) -> Vec<f32> {
    let radius = (k.len() / 2) as isize;
    let mut tmp = vec![0.0f32; w * h];
    // Horizontal.
    for y in 0..h {
        for xpix in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sx = xpix as isize + ki as isize - radius;
                if sx >= 0 && (sx as usize) < w {
                    acc += kv * src[y * w + sx as usize];
                }
            }
            tmp[y * w + xpix] = acc;
        }
    }
    // Vertical.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for xpix in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sy = y as isize + ki as isize - radius;
                if sy >= 0 && (sy as usize) < h {
                    acc += kv * tmp[sy as usize * w + xpix];
                }
            }
            out[y * w + xpix] = acc;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_target(w: usize, h: usize, color: [f32; 3]) -> TargetImage {
        let mut data = vec![0.0f32; w * h * 4];
        for px in data.chunks_exact_mut(4) {
            px[0] = color[0];
            px[1] = color[1];
            px[2] = color[2];
            px[3] = 1.0;
        }
        TargetImage { width: w, height: h, data }
    }

    fn test_camera(w: u32, h: u32) -> Camera {
        Camera {
            width: w,
            height: h,
            fx: w as f32,
            fy: w as f32,
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
            // Camera at origin looking down +z (OpenCV); c2w = identity.
            c2w: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            distortion: [0.0; 3],
            image_path: None,
        }
    }

    #[test]
    fn ssim_of_identical_images_is_one() {
        let w = 16;
        let h = 16;
        let img: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32 / 7.0).collect();
        let k = gaussian_kernel_1d(5, 1.5);
        let (s, _) = ssim_channel(&img, &img, w, h, &k);
        assert!((s / (w * h) as f32 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn image_loss_gradient_matches_finite_difference() {
        let (w, h) = (12usize, 12usize);
        // Pseudo-random but deterministic rendered & target buffers.
        let mut rendered = vec![0.0f32; w * h * 3];
        let mut target = vec![0.0f32; w * h * 4];
        for p in 0..w * h {
            for ch in 0..3 {
                rendered[p * 3 + ch] = ((p * 3 + ch) as f32 * 0.123).sin() * 0.4 + 0.5;
                target[p * 4 + ch] = ((p * 5 + ch) as f32 * 0.077).cos() * 0.4 + 0.5;
            }
            target[p * 4 + 3] = 1.0;
        }
        let lambda = 0.2;
        let (_, grad) = photometric_loss_grad(&rendered, &target, w, h, lambda);

        let eps = 1e-3;
        // Check a handful of pixels/channels (interior and edge).
        for &(p, ch) in &[(0usize, 0usize), (40, 1), (77, 2), (143, 0)] {
            let idx = p * 3 + ch;
            let mut rp = rendered.clone();
            let mut rm = rendered.clone();
            rp[idx] += eps;
            rm[idx] -= eps;
            let lp = photometric_loss_grad(&rp, &target, w, h, lambda).0;
            let lm = photometric_loss_grad(&rm, &target, w, h, lambda).0;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (grad[idx] - num).abs() < 2e-3,
                "loss grad mismatch at ({p},{ch}): analytic {} vs numeric {}",
                grad[idx],
                num
            );
        }
    }

    #[test]
    fn render_produces_nonblack_for_visible_gaussian() {
        let cam = test_camera(32, 32);
        let mut g = Gaussian3D::new([0.0, 0.0, 2.0]);
        g.opacity_logit = crate::math::logit(0.9);
        g.log_scale = [-1.5, -1.5, -1.5];
        g.sh_coeffs[0] = 2.0; // bright-ish red DC
        let img = render_color(&[g], &cam, [0.0, 0.0, 0.0]);
        let center = (16 * 32 + 16) * 3;
        assert!(img[center] > 0.01, "center pixel should be lit: {}", img[center]);
    }

    #[test]
    fn parameter_gradients_match_finite_difference() {
        let cam = test_camera(24, 24);
        let bg = [0.0, 0.0, 0.0];
        // L1-only against a black target ⇒ the loss is locally smooth (no sign
        // flips), giving clean central differences to validate against.
        let cfg = LossConfig { lambda_dssim: 0.0 };
        let target = tiny_target(24, 24, [0.0, 0.0, 0.0]);

        // A single, comfortably visible Gaussian with view-dependent color that
        // stays within (0, 1) so no channel saturates.
        let mut g = Gaussian3D::new([0.1, -0.05, 2.0]);
        g.opacity_logit = crate::math::logit(0.6);
        g.log_scale = [-2.2, -2.0, -2.1];
        g.rotation = crate::math::quaternion::normalize([0.9, 0.1, -0.2, 0.15]);
        for ch in 0..3 {
            g.sh_coeffs[ch * 16] = 0.4; // positive DC
            for k in 1..16 {
                g.sh_coeffs[ch * 16 + k] = ((ch * 16 + k) as f32 * 0.05).sin() * 0.04;
            }
        }
        let gauss = vec![g];

        let (_, grads) = render_and_backward(&gauss, &cam, &target, bg, &cfg);
        let grad = &grads[0];

        let loss_at = |gs: &[Gaussian3D]| render_and_backward(gs, &cam, &target, bg, &cfg).0;
        let eps = 1e-3;
        let numeric = |perturb: &dyn Fn(&mut Gaussian3D, f32)| -> f32 {
            let mut gp = gauss.clone();
            let mut gm = gauss.clone();
            perturb(&mut gp[0], eps);
            perturb(&mut gm[0], -eps);
            (loss_at(&gp) - loss_at(&gm)) / (2.0 * eps)
        };
        let check = |analytic: f32, num: f32, name: &str| {
            let denom = analytic.abs().max(num.abs()).max(1e-4);
            let rel = (analytic - num).abs() / denom;
            assert!(
                rel < 0.05 || (analytic - num).abs() < 3e-4,
                "{name}: analytic {analytic} vs numeric {num} (rel {rel})"
            );
        };

        for k in 0..3 {
            check(grad.d_position[k], numeric(&move |g, e| g.position[k] += e), "d_position");
        }
        check(grad.d_opacity_logit, numeric(&|g, e| g.opacity_logit += e), "d_opacity");
        for k in 0..3 {
            check(grad.d_log_scale[k], numeric(&move |g, e| g.log_scale[k] += e), "d_log_scale");
        }
        for &k in &[0usize, 1, 16, 33] {
            check(grad.d_sh[k], numeric(&move |g, e| g.sh_coeffs[k] += e), "d_sh");
        }

        // Rotation: the forward renormalizes the quaternion, so only the
        // component tangent to q is meaningful. Compare tangential projections.
        let q = gauss[0].rotation;
        let mut num_rot = [0.0f32; 4];
        for k in 0..4 {
            num_rot[k] = numeric(&move |g, e| g.rotation[k] += e);
        }
        let proj_tangent = |v: [f32; 4]| {
            let dot = v[0] * q[0] + v[1] * q[1] + v[2] * q[2] + v[3] * q[3];
            [
                v[0] - dot * q[0],
                v[1] - dot * q[1],
                v[2] - dot * q[2],
                v[3] - dot * q[3],
            ]
        };
        let a = proj_tangent(grad.d_rotation);
        let n = proj_tangent(num_rot);
        let diff: f32 = (0..4).map(|i| (a[i] - n[i]).powi(2)).sum::<f32>().sqrt();
        let mag: f32 = (0..4).map(|i| n[i] * n[i]).sum::<f32>().sqrt().max(1e-3);
        assert!(diff / mag < 0.1, "rotation tangential grad: analytic {a:?} vs numeric {n:?}");
    }
}
