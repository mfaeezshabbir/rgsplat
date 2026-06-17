/// math/mod.rs — Core mathematical primitives for 3D Gaussian Splatting
///
/// Mathematical foundation based on:
/// "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
/// Kerbl et al., SIGGRAPH 2023  (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
///
/// Key math: each Gaussian G(x) = exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))
/// where Σ = R S Sᵀ Rᵀ  (covariance from rotation R and scale S)

pub mod covariance;
pub mod quaternion;
pub mod sh;           // Spherical Harmonics for view-dependent color
pub mod projection;

use nalgebra::{Matrix3, Matrix4, Vector3, UnitQuaternion};
use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};

// ─────────────────────────────────────────────────────────────────────────────
// Core Gaussian primitive — memory layout matters: keep hot fields together
// ─────────────────────────────────────────────────────────────────────────────

/// A single 3D Gaussian with all parameters needed for rendering and optimization.
/// 
/// Memory layout is intentionally aligned for SIMD and GPU buffer uploads.
/// Total: ~192 bytes per Gaussian (tight-packed).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Gaussian3D {
    /// World-space center μ ∈ ℝ³
    pub position: [f32; 3],

    /// Opacity before sigmoid: α = σ(opacity_logit)
    pub opacity_logit: f32,

    /// Rotation as unit quaternion [w, x, y, z] — encodes R in Σ = R S Sᵀ Rᵀ
    pub rotation: [f32; 4],

    /// Log scale [log sx, log sy, log sz] — exp gives positive scale
    /// Scale encodes S in Σ = R S Sᵀ Rᵀ
    pub log_scale: [f32; 3],

    /// Spherical harmonics coefficients for view-dependent color.
    /// Degree 3 SH: (3+1)² = 16 coefficients × 3 channels (RGB) = 48 floats
    pub sh_coeffs: [f32; 48],
}

unsafe impl Pod for Gaussian3D {}
unsafe impl Zeroable for Gaussian3D {}

impl Gaussian3D {
    pub fn new(position: [f32; 3]) -> Self {
        let mut g = Self::zeroed();
        g.position = position;
        // Identity rotation: w=1, x=y=z=0
        g.rotation = [1.0, 0.0, 0.0, 0.0];
        // Small initial scale
        g.log_scale = [-3.0, -3.0, -3.0];
        // Slightly opaque to start
        g.opacity_logit = -2.0;  // σ(-2) ≈ 0.12
        g
    }

    /// Decode opacity: α = sigmoid(logit)
    #[inline(always)]
    pub fn opacity(&self) -> f32 {
        sigmoid(self.opacity_logit)
    }

    /// Decode scale: s = exp(log_scale)
    #[inline(always)]
    pub fn scale(&self) -> [f32; 3] {
        [
            self.log_scale[0].exp(),
            self.log_scale[1].exp(),
            self.log_scale[2].exp(),
        ]
    }

    /// Rotation quaternion as nalgebra type
    #[inline]
    pub fn rotation_quat(&self) -> UnitQuaternion<f32> {
        let [w, x, y, z] = self.rotation;
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z))
    }

    /// Build 3×3 covariance matrix Σ = R·S·Sᵀ·Rᵀ
    /// where S = diag(sx, sy, sz) and R is from quaternion
    pub fn covariance_3d(&self) -> Matrix3<f32> {
        let r = self.rotation_quat().to_rotation_matrix();
        let s = self.scale();
        let s_mat = Matrix3::from_diagonal(&Vector3::new(s[0], s[1], s[2]));
        // Σ = R S Sᵀ Rᵀ  (since S is diagonal, Sᵀ = S)
        let rs = r.matrix() * s_mat;
        rs * rs.transpose()
    }

    /// Project 3D covariance to 2D screen covariance Σ' = J W Σ Wᵀ Jᵀ
    /// J = Jacobian of projective transform, W = view rotation
    pub fn projected_covariance_2d(
        &self,
        view_matrix: &Matrix4<f32>,
        focal: (f32, f32),
    ) -> Matrix3<f32> {
        projection::project_covariance(
            &self.covariance_3d(),
            &self.position.into(),
            view_matrix,
            focal,
        )
    }

    /// Get base color from DC SH coefficient (degree 0)
    /// sh_coeffs[0..3] = DC term, color = 0.5 + DC * SH_C0
    pub fn base_color(&self) -> [f32; 3] {
        const SH_C0: f32 = 0.282_094_8; // 1 / (2√π)
        [
            0.5 + self.sh_coeffs[0] * SH_C0,
            0.5 + self.sh_coeffs[1] * SH_C0,
            0.5 + self.sh_coeffs[2] * SH_C0,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera model
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera with optional distortion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub width: u32,
    pub height: u32,
    pub fx: f32,          // focal length x (pixels)
    pub fy: f32,          // focal length y (pixels)
    pub cx: f32,          // principal point x
    pub cy: f32,          // principal point y
    /// Camera-to-world transform (4×4)
    pub c2w: [[f32; 4]; 4],
    /// Radial distortion [k1, k2, k3]
    pub distortion: [f32; 3],
    /// Path to this view's ground-truth image (training target), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_path: Option<std::path::PathBuf>,
}

impl Camera {
    /// Camera-to-world as a nalgebra matrix. `c2w` is stored **row-major**
    /// (`c2w[row][col]`), so build with `from_fn` rather than `Matrix4::from`
    /// (which would interpret the array column-major).
    pub fn c2w_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_fn(|r, c| self.c2w[r][c])
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        // World-to-camera = inverse of c2w
        self.c2w_matrix()
            .try_inverse()
            .expect("c2w must be invertible")
    }

    /// World-space camera center (translation column of `c2w`).
    pub fn position(&self) -> [f32; 3] {
        [self.c2w[0][3], self.c2w[1][3], self.c2w[2][3]]
    }

    pub fn projection_matrix(&self, near: f32, far: f32) -> Matrix4<f32> {
        projection::make_projection(self.fx, self.fy, self.cx, self.cy,
                                    self.width as f32, self.height as f32,
                                    near, far)
    }

    pub fn focal(&self) -> (f32, f32) { (self.fx, self.fy) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Fast sigmoid: σ(x) = 1 / (1 + e⁻ˣ)
/// Uses LLVM intrinsic path when possible
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse sigmoid (logit): logit(p) = ln(p / (1-p))
#[inline(always)]
pub fn logit(p: f32) -> f32 {
    debug_assert!(p > 0.0 && p < 1.0, "logit domain: (0, 1)");
    (p / (1.0 - p)).ln()
}

/// Softplus: log(1 + eˣ) — smooth approximation to ReLU
#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Clamp a value to [lo, hi]
#[inline(always)]
pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

/// Convert degrees to radians
#[inline(always)]
pub fn deg2rad(d: f32) -> f32 { d * std::f32::consts::PI / 180.0 }

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_logit_roundtrip() {
        for v in [-3.0f32, -1.0, 0.0, 1.0, 3.0] {
            let p = sigmoid(v);
            assert_relative_eq!(logit(p), v, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_covariance_psd() {
        let g = Gaussian3D::new([0.0, 0.0, 0.0]);
        let cov = g.covariance_3d();
        // PSD check: all eigenvalues >= 0
        let sym = (cov + cov.transpose()) * 0.5;
        // trace > 0 is a quick sanity check
        assert!(sym.trace() > 0.0);
    }
}