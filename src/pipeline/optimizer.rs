/// pipeline/optimizer.rs — Adam optimizer + densification for Gaussian training
///
/// Implements the training loop from the 3DGS paper:
///   - Adam optimizer for position, rotation, scale, opacity, SH coefficients
///   - Adaptive Gaussian densification (split + clone)
///   - Opacity pruning
///   - Per-parameter learning rates
///
/// Reference: Algorithm 1 in Kerbl et al. 2023

use rayon::prelude::*;
use tracing::info;

use crate::math::{Gaussian3D, logit};

pub use crate::cpu::backward::GaussianGrad;

// ─────────────────────────────────────────────────────────────────────────────
// Hyper-parameters
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub num_iterations:        usize,
    pub lr_position:           f32,   // 0.00016
    pub lr_rotation:           f32,   // 0.001
    pub lr_scale:              f32,   // 0.005
    pub lr_opacity:            f32,   // 0.05
    pub lr_sh_dc:              f32,   // 0.0025
    pub lr_sh_rest:            f32,   // 0.000125  (= lr_sh_dc / 20)
    pub beta1:                 f32,   // 0.9
    pub beta2:                 f32,   // 0.999
    pub eps:                   f32,   // 1e-8
    pub densify_from_iter:     usize, // 500
    pub densify_until_iter:    usize, // 15000
    pub densify_grad_threshold: f32,  // 0.0002
    pub opacity_reset_iter:    usize, // 3000
    pub prune_opacity_thresh:  f32,   // 0.005
    pub max_gaussians:         usize, // 3_000_000
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_iterations: 30_000,
            lr_position: 0.00016,
            lr_rotation: 0.001,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_sh_dc: 0.0025,
            lr_sh_rest: 0.000125,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            densify_from_iter: 500,
            densify_until_iter: 15_000,
            densify_grad_threshold: 0.0002,
            opacity_reset_iter: 3000,
            prune_opacity_thresh: 0.005,
            max_gaussians: 3_000_000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-Gaussian Adam state
// ─────────────────────────────────────────────────────────────────────────────

/// Number of trainable floats per Gaussian:
/// position(3) + rotation(4) + scale(3) + opacity(1) + sh(48) = 59
const PARAMS_PER_GAUSSIAN: usize = 59;

/// Adam optimizer state for all Gaussians.
pub struct AdamState {
    /// First moment (mean of gradients): shape [N × PARAMS_PER_GAUSSIAN]
    pub m1: Vec<f32>,
    /// Second moment (variance of gradients)
    pub m2: Vec<f32>,
    /// Accumulated 2D gradient magnitude (for densification)
    pub grad_accum: Vec<f32>,
    pub grad_denom: Vec<u32>,
    /// Current iteration (for bias correction)
    pub t: usize,
}

impl AdamState {
    pub fn new(n: usize) -> Self {
        Self {
            m1: vec![0.0; n * PARAMS_PER_GAUSSIAN],
            m2: vec![0.0; n * PARAMS_PER_GAUSSIAN],
            grad_accum: vec![0.0; n],
            grad_denom: vec![0u32; n],
            t: 0,
        }
    }

    pub fn resize(&mut self, new_n: usize) {
        let old_n = self.m1.len() / PARAMS_PER_GAUSSIAN;
        if new_n > old_n {
            self.m1.resize(new_n * PARAMS_PER_GAUSSIAN, 0.0);
            self.m2.resize(new_n * PARAMS_PER_GAUSSIAN, 0.0);
            self.grad_accum.resize(new_n, 0.0);
            self.grad_denom.resize(new_n, 0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam update step
// ─────────────────────────────────────────────────────────────────────────────

/// Apply one Adam step for all Gaussians.
///
/// Called after backward pass (gradients computed).
/// Parallel over Gaussians with Rayon.
pub fn adam_step(
    gaussians: &mut [Gaussian3D],
    grads: &[GaussianGrad],
    state: &mut AdamState,
    cfg: &TrainConfig,
) {
    state.t += 1;

    // Bias correction factors
    let bc1 = 1.0 - cfg.beta1.powi(state.t as i32);
    let bc2 = 1.0 - cfg.beta2.powi(state.t as i32);
    let lrs = learning_rates(cfg);

    // Each Gaussian owns a disjoint slice of m1/m2/grad_accum/grad_denom, so we
    // can parallelize without aliasing by zipping `par_chunks_mut`.
    gaussians
        .par_iter_mut()
        .zip(grads.par_iter())
        .zip(state.m1.par_chunks_mut(PARAMS_PER_GAUSSIAN))
        .zip(state.m2.par_chunks_mut(PARAMS_PER_GAUSSIAN))
        .zip(state.grad_accum.par_iter_mut())
        .zip(state.grad_denom.par_iter_mut())
        .for_each(|(((((g, grad), m1), m2), accum), denom)| {
            let flat_grad = flatten_grad(grad);

            for (j, (gj, lrj)) in flat_grad.iter().zip(lrs.iter()).enumerate() {
                m1[j] = cfg.beta1 * m1[j] + (1.0 - cfg.beta1) * gj;
                m2[j] = cfg.beta2 * m2[j] + (1.0 - cfg.beta2) * gj * gj;
                let m1_hat = m1[j] / bc1;
                let m2_hat = m2[j] / bc2;
                let update = lrj * m1_hat / (m2_hat.sqrt() + cfg.eps);
                apply_update_to_gaussian(g, j, update);
            }

            // Accumulate screen-space gradient for densification.
            *accum += grad.screen_grad_norm;
            *denom += 1;
        });

    // Normalize quaternion (rotation must stay unit)
    gaussians.par_iter_mut().for_each(|g| {
        let q = &mut g.rotation;
        let norm = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
        if norm > 1e-8 {
            q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm;
        }
    });
}

fn flatten_grad(g: &GaussianGrad) -> Vec<f32> {
    let mut v = Vec::with_capacity(PARAMS_PER_GAUSSIAN);
    v.extend_from_slice(&g.d_position);
    v.extend_from_slice(&g.d_rotation);
    v.extend_from_slice(&g.d_log_scale);
    v.push(g.d_opacity_logit);
    v.extend_from_slice(&g.d_sh);
    v
}

fn learning_rates(cfg: &TrainConfig) -> Vec<f32> {
    let mut lr = Vec::with_capacity(PARAMS_PER_GAUSSIAN);
    lr.extend(std::iter::repeat(cfg.lr_position).take(3));  // position
    lr.extend(std::iter::repeat(cfg.lr_rotation).take(4));  // rotation
    lr.extend(std::iter::repeat(cfg.lr_scale).take(3));     // scale
    lr.push(cfg.lr_opacity);                                  // opacity
    lr.extend(std::iter::repeat(cfg.lr_sh_dc).take(3));     // SH DC
    lr.extend(std::iter::repeat(cfg.lr_sh_rest).take(45));  // SH higher
    lr
}

fn apply_update_to_gaussian(g: &mut Gaussian3D, param_idx: usize, update: f32) {
    match param_idx {
        0..=2 => g.position[param_idx] -= update,
        3..=6 => g.rotation[param_idx - 3] -= update,
        7..=9 => g.log_scale[param_idx - 7] -= update,
        10    => g.opacity_logit -= update,
        11..=58 => g.sh_coeffs[param_idx - 11] -= update,
        _ => unreachable!(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Densification (split + clone) and pruning
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive densification and pruning.
///
/// Clone: small Gaussians in under-reconstructed regions (high 2D gradient)
/// Split: large Gaussians into 2 smaller ones
/// Prune: transparent or too-large Gaussians
pub fn densify_and_prune(
    gaussians: &mut Vec<Gaussian3D>,
    state: &mut AdamState,
    cfg: &TrainConfig,
    scene_extent: f32,
) {
    let n = gaussians.len();
    let mut to_add: Vec<Gaussian3D> = Vec::new();
    let mut to_remove: Vec<bool> = vec![false; n];

    // Compute average screen gradient per Gaussian
    let avg_grads: Vec<f32> = (0..n).map(|i| {
        if state.grad_denom[i] > 0 {
            state.grad_accum[i] / state.grad_denom[i] as f32
        } else {
            0.0
        }
    }).collect();

    for i in 0..n {
        let g = &gaussians[i];
        let alpha = g.opacity();
        let scale = g.scale();
        let max_scale = scale[0].max(scale[1]).max(scale[2]);

        // Prune: too transparent or too large
        if alpha < cfg.prune_opacity_thresh || max_scale > 0.1 * scene_extent {
            to_remove[i] = true;
            continue;
        }

        if avg_grads[i] >= cfg.densify_grad_threshold
            && gaussians.len() + to_add.len() < cfg.max_gaussians
        {
            if max_scale > 0.01 * scene_extent {
                // Split: replace with 2 smaller Gaussians
                to_remove[i] = true;
                let children = split_gaussian(g);
                to_add.extend(children);
            } else {
                // Clone: add a copy (slightly displaced)
                to_add.push(clone_gaussian(g, avg_grads[i]));
            }
        }
    }

    // Apply removals (compact array)
    let mut j = 0;
    gaussians.retain(|_| {
        let keep = !to_remove[j];
        j += 1;
        keep
    });

    let kept = gaussians.len();
    gaussians.extend(to_add);
    state.resize(gaussians.len());

    // Reset gradient accumulators for next densification window
    for acc in &mut state.grad_accum { *acc = 0.0; }
    for den in &mut state.grad_denom { *den = 0; }

    info!(
        "Densification: kept={}, removed={}, total={}",
        kept, n - kept, gaussians.len()
    );
}

fn split_gaussian(g: &Gaussian3D) -> [Gaussian3D; 2] {
    use nalgebra::Vector3;

    let scale = g.scale();
    let r = g.rotation_quat();

    // Sample two positions along the major axis
    let major_axis = r * Vector3::new(scale[0], 0.0, 0.0);
    let offset = major_axis.normalize() * scale[0];

    let mut g1 = g.clone();
    let mut g2 = g.clone();

    g1.position[0] += offset.x; g1.position[1] += offset.y; g1.position[2] += offset.z;
    g2.position[0] -= offset.x; g2.position[1] -= offset.y; g2.position[2] -= offset.z;

    // Reduce scale by √2 in split axis
    let new_scale = (scale[0] / std::f32::consts::SQRT_2).max(1e-6).ln();
    g1.log_scale[0] = new_scale;
    g2.log_scale[0] = new_scale;

    // Reduce opacity slightly (two splats cover same area)
    let new_opacity = (g.opacity() * 0.8).clamp(1e-6, 0.999);
    g1.opacity_logit = logit(new_opacity);
    g2.opacity_logit = logit(new_opacity);

    [g1, g2]
}

fn clone_gaussian(g: &Gaussian3D, grad_strength: f32) -> Gaussian3D {
    // Clone with tiny random perturbation in gradient direction
    // (In production: use the actual 2D gradient direction from backward pass)
    let mut clone = g.clone();
    let jitter = grad_strength * 0.01;
    clone.position[0] += jitter;
    clone.position[1] += jitter;
    clone
}

/// Reset opacities to near-zero periodically (forces re-learning of opacity).
pub fn reset_opacities(gaussians: &mut [Gaussian3D]) {
    const RESET_VALUE: f32 = 0.01;
    gaussians.par_iter_mut().for_each(|g| {
        g.opacity_logit = logit(RESET_VALUE.min(g.opacity()));
    });
}