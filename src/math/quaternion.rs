//! math/quaternion.rs — Quaternion helpers for Gaussian rotations.
//!
//! Convention: quaternions are stored `[w, x, y, z]` (scalar-first), matching
//! both COLMAP (Hamilton convention) and the 3DGS `.ply` `rot_0..3` layout.

use nalgebra::Matrix3;

/// Normalize a `[w, x, y, z]` quaternion. Returns identity on a (near-)zero input.
#[inline]
pub fn normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// The quaternion→matrix algebraic formula evaluated at the given components
/// (no normalization). For a unit quaternion this is a rotation matrix; the
/// analytic [`rotation_jacobian`] differentiates exactly this expression.
#[inline]
pub fn mat_from_quat_components(q: [f32; 4]) -> Matrix3<f32> {
    let [w, x, y, z] = q;
    Matrix3::new(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y),
    )
}

/// Convert a (not necessarily normalized) `[w, x, y, z]` quaternion to a 3×3
/// rotation matrix. The quaternion is normalized internally.
#[inline]
pub fn to_rotation_matrix(q: [f32; 4]) -> Matrix3<f32> {
    mat_from_quat_components(normalize(q))
}

/// Partial derivatives `∂R/∂q_k` for k in {w, x, y, z}, evaluated at a
/// **normalized** quaternion. Returns four 3×3 matrices in `[w, x, y, z]` order.
///
/// Used by the analytic covariance backward pass. The radial (normalization)
/// component is intentionally omitted — quaternions are renormalized after each
/// optimizer step, which removes it.
pub fn rotation_jacobian(q: [f32; 4]) -> [Matrix3<f32>; 4] {
    let [w, x, y, z] = normalize(q);

    let dr_dw = Matrix3::new(
        0.0, -2.0 * z, 2.0 * y,
        2.0 * z, 0.0, -2.0 * x,
        -2.0 * y, 2.0 * x, 0.0,
    );
    let dr_dx = Matrix3::new(
        0.0, 2.0 * y, 2.0 * z,
        2.0 * y, -4.0 * x, -2.0 * w,
        2.0 * z, 2.0 * w, -4.0 * x,
    );
    let dr_dy = Matrix3::new(
        -4.0 * y, 2.0 * x, 2.0 * w,
        2.0 * x, 0.0, 2.0 * z,
        -2.0 * w, 2.0 * z, -4.0 * y,
    );
    let dr_dz = Matrix3::new(
        -4.0 * z, -2.0 * w, 2.0 * x,
        2.0 * w, -4.0 * z, 2.0 * y,
        2.0 * x, 2.0 * y, 0.0,
    );

    [dr_dw, dr_dx, dr_dy, dr_dz]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_is_identity() {
        let r = to_rotation_matrix([1.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(r, Matrix3::identity(), epsilon = 1e-6);
    }

    #[test]
    fn rotation_is_orthonormal() {
        let r = to_rotation_matrix([0.5, 0.5, 0.5, 0.5]);
        let should_be_identity = r * r.transpose();
        assert_relative_eq!(should_be_identity, Matrix3::identity(), epsilon = 1e-5);
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        // The analytic jacobian differentiates the raw quat→matrix formula, so
        // finite-difference that same (unnormalized) formula at a unit quaternion.
        let q = normalize([0.3, 0.7, -0.2, 0.5]);
        let analytic = rotation_jacobian(q);
        let eps = 1e-4;
        for k in 0..4 {
            let mut qp = q;
            let mut qm = q;
            qp[k] += eps;
            qm[k] -= eps;
            let num = (mat_from_quat_components(qp) - mat_from_quat_components(qm)) / (2.0 * eps);
            let diff = (analytic[k] - num).norm();
            assert!(diff < 1e-2, "q component {k}: jacobian diff {diff}");
        }
    }
}
