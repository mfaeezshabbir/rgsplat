//! math/covariance.rs — 3D covariance construction and its gradients.
//!
//! A Gaussian's world-space covariance is `Σ = R S Sᵀ Rᵀ = M Mᵀ` with `M = R S`,
//! where `R` comes from the rotation quaternion and `S = diag(s)` from the scale.
//!
//! This module provides both the forward construction and the backward pass that
//! turns `∂L/∂Σ` (a symmetric 3×3 matrix) into gradients on `log_scale` and the
//! rotation quaternion — the pieces the optimizer actually updates.

use nalgebra::Matrix3;

use super::quaternion;

/// Build `M = R · S` (the matrix square-root factor of the covariance).
#[inline]
pub fn rs_matrix(scale: [f32; 3], rot: [f32; 4]) -> Matrix3<f32> {
    let r = quaternion::to_rotation_matrix(rot);
    // R * diag(s) scales column j by s[j].
    Matrix3::new(
        r[(0, 0)] * scale[0], r[(0, 1)] * scale[1], r[(0, 2)] * scale[2],
        r[(1, 0)] * scale[0], r[(1, 1)] * scale[1], r[(1, 2)] * scale[2],
        r[(2, 0)] * scale[0], r[(2, 1)] * scale[1], r[(2, 2)] * scale[2],
    )
}

/// Build the world-space covariance `Σ = R S Sᵀ Rᵀ`.
#[inline]
pub fn build_cov3d(scale: [f32; 3], rot: [f32; 4]) -> Matrix3<f32> {
    let m = rs_matrix(scale, rot);
    m * m.transpose()
}

/// Backpropagate `∂L/∂Σ` to `∂L/∂log_scale` and `∂L/∂q`.
///
/// # Arguments
/// * `scale`   — decoded (linear) scale `s = exp(log_scale)`
/// * `rot`     — rotation quaternion `[w, x, y, z]`
/// * `dl_dsigma` — gradient of the loss w.r.t. the (symmetric) covariance `Σ`
///
/// # Returns
/// `(d_log_scale[3], d_rot[4])`.
pub fn backprop_cov3d(
    scale: [f32; 3],
    rot: [f32; 4],
    dl_dsigma: &Matrix3<f32>,
) -> ([f32; 3], [f32; 4]) {
    let r = quaternion::to_rotation_matrix(rot);
    let m = rs_matrix(scale, rot); // M = R S

    // Σ = M Mᵀ  ⇒  dL/dM = (G + Gᵀ) M, with G = dL/dΣ.
    let g = dl_dsigma;
    let g_sym = g + g.transpose();
    let dl_dm = g_sym * m;

    // M[i,j] = R[i,j] * s[j].
    //   dL/ds[j]     = Σ_i dL/dM[i,j] * R[i,j]
    //   dL/dR[i,j]   = dL/dM[i,j] * s[j]
    let mut dl_dscale = [0.0f32; 3];
    let mut dl_dr = Matrix3::zeros();
    for j in 0..3 {
        let mut acc = 0.0;
        for i in 0..3 {
            acc += dl_dm[(i, j)] * r[(i, j)];
            dl_dr[(i, j)] = dl_dm[(i, j)] * scale[j];
        }
        dl_dscale[j] = acc;
    }

    // log-scale: s = exp(log_scale) ⇒ dL/dlog_scale = dL/ds * s.
    let d_log_scale = [
        dl_dscale[0] * scale[0],
        dl_dscale[1] * scale[1],
        dl_dscale[2] * scale[2],
    ];

    // Rotation: contract dL/dR with each ∂R/∂q_k.
    let jac = quaternion::rotation_jacobian(rot);
    let mut d_rot = [0.0f32; 4];
    for k in 0..4 {
        d_rot[k] = jac[k].component_mul(&dl_dr).sum();
    }

    (d_log_scale, d_rot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cov_is_symmetric_psd() {
        let cov = build_cov3d([0.3, 0.2, 0.5], [0.5, 0.5, 0.5, 0.5]);
        assert_relative_eq!(cov, cov.transpose(), epsilon = 1e-6);
        assert!(cov.determinant() > 0.0);
    }

    #[test]
    fn scale_gradient_matches_finite_difference() {
        // Pick an arbitrary scalar loss L = <W, Σ> for a fixed weight matrix W,
        // so dL/dΣ = W and we can finite-difference w.r.t. log_scale.
        let w = Matrix3::new(0.7, 0.1, -0.2, 0.1, 0.4, 0.3, -0.2, 0.3, 0.9);
        let log_scale = [-1.0f32, -0.5, 0.2];
        let rot = quaternion::normalize([0.4, 0.2, -0.7, 0.5]);
        let scale = [log_scale[0].exp(), log_scale[1].exp(), log_scale[2].exp()];

        let (d_log_scale, _) = backprop_cov3d(scale, rot, &w);

        let eps = 1e-3;
        for k in 0..3 {
            let mut lp = log_scale;
            let mut lm = log_scale;
            lp[k] += eps;
            lm[k] -= eps;
            let sp = [lp[0].exp(), lp[1].exp(), lp[2].exp()];
            let sm = [lm[0].exp(), lm[1].exp(), lm[2].exp()];
            let loss = |s: [f32; 3]| w.component_mul(&build_cov3d(s, rot)).sum();
            let num = (loss(sp) - loss(sm)) / (2.0 * eps);
            assert_relative_eq!(d_log_scale[k], num, epsilon = 1e-2);
        }
    }
}
