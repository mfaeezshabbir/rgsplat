/// math/projection.rs — Projective geometry for Gaussian Splatting
///
/// Implements EWA (Elliptical Weighted Average) splatting projection:
/// Σ'₂D = J · W · Σ₃D · Wᵀ · Jᵀ   (Zwicker et al. 2001)
///
/// Where:
///   W = upper-left 3×3 of view matrix (rotation only)
///   J = affine approximation Jacobian of projective transform
///   Σ₃D = 3D world-space covariance

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};

/// Project 3D Gaussian covariance to 2D screen covariance.
///
/// # Arguments
/// * `sigma_3d`    — 3×3 world covariance Σ
/// * `position`    — Gaussian center in world space
/// * `view`        — 4×4 view (world-to-camera) matrix
/// * `focal`       — (fx, fy) focal lengths in pixels
///
/// # Returns
/// 3×3 matrix — but only the upper-left 2×2 sub-block is the 2D covariance.
/// Third row/col carry the z component for depth culling.
pub fn project_covariance(
    sigma_3d: &Matrix3<f32>,
    position: &[f32; 3],
    view: &Matrix4<f32>,
    focal: (f32, f32),
) -> Matrix3<f32> {
    // Transform center to camera space
    let p_world = Vector4::new(position[0], position[1], position[2], 1.0);
    let p_cam = view * p_world;
    let (tx, ty, tz) = (p_cam.x, p_cam.y, p_cam.z);

    // Clamp tz to avoid division by zero / behind-camera issues
    let tz = tz.max(0.001);

    let (fx, fy) = focal;

    // Jacobian of the projective mapping p → (fx·px/pz, fy·py/pz)
    //
    //       [ fx/tz    0    -fx·tx/tz² ]
    //  J =  [  0     fy/tz  -fy·ty/tz² ]
    //       [  0       0        0       ]   ← z row dropped for 2D
    //
    // We keep it as 3×3 for the matrix multiply, then trim.
    let j = Matrix3::new(
        fx / tz,   0.0,      -fx * tx / (tz * tz),
        0.0,       fy / tz,  -fy * ty / (tz * tz),
        0.0,       0.0,       0.0,
    );

    // Extract view rotation W (upper-left 3×3 of view matrix)
    let w = Matrix3::new(
        view[(0, 0)], view[(0, 1)], view[(0, 2)],
        view[(1, 0)], view[(1, 1)], view[(1, 2)],
        view[(2, 0)], view[(2, 1)], view[(2, 2)],
    );

    // T = J · W  →  Σ'₂D = T · Σ₃D · Tᵀ
    let t = j * w;
    let sigma_2d = t * sigma_3d * t.transpose();

    // Add a small "low-pass" filter (0.3px) to avoid aliasing at tiny scales
    let mut result = sigma_2d;
    result[(0, 0)] += 0.3;
    result[(1, 1)] += 0.3;
    result
}

/// Build OpenGL-style perspective projection matrix for a pinhole camera.
///
/// Produces a matrix that maps view-space → NDC in range [-1, 1].
/// Compatible with wgpu / Vulkan with y-flip applied separately.
pub fn make_projection(
    fx: f32, fy: f32,
    cx: f32, cy: f32,
    width: f32, height: f32,
    near: f32, far: f32,
) -> Matrix4<f32> {
    // OpenCV → OpenGL convention
    let (w, h) = (width, height);

    Matrix4::new(
        2.0 * fx / w,    0.0,           (w - 2.0 * cx) / w,  0.0,
        0.0,             -2.0 * fy / h, (h - 2.0 * cy) / h,  0.0,
        0.0,              0.0,          -(far + near) / (far - near),  -(2.0 * far * near) / (far - near),
        0.0,              0.0,          -1.0,                  0.0,
    )
}

/// Compute view frustum planes (for Gaussian culling).
/// Returns 6 planes as (normal: [f32;3], distance: f32).
pub fn frustum_planes(proj_view: &Matrix4<f32>) -> [(Vector3<f32>, f32); 6] {
    // Gribb-Hartmann fast frustum extraction from MVP matrix rows
    let m = proj_view;
    let planes_raw = [
        // left:   row3 + row0
        ( m.row(3) + m.row(0) ),
        // right:  row3 - row0
        ( m.row(3) - m.row(0) ),
        // bottom: row3 + row1
        ( m.row(3) + m.row(1) ),
        // top:    row3 - row1
        ( m.row(3) - m.row(1) ),
        // near:   row3 + row2
        ( m.row(3) + m.row(2) ),
        // far:    row3 - row2
        ( m.row(3) - m.row(2) ),
    ];

    planes_raw.map(|row| {
        let n = Vector3::new(row[0], row[1], row[2]);
        let len = n.norm();
        (n / len, row[3] / len)
    })
}

/// Fast sphere-frustum intersection test.
/// Returns false if the sphere is fully outside any frustum plane.
#[inline]
pub fn sphere_in_frustum(
    center: &[f32; 3],
    radius: f32,
    planes: &[(Vector3<f32>, f32); 6],
) -> bool {
    let p = Vector3::new(center[0], center[1], center[2]);
    for (normal, dist) in planes {
        if normal.dot(&p) + dist < -radius {
            return false;  // outside this plane → cull
        }
    }
    true
}

/// Compute the 2D axis-aligned bounding box of a projected Gaussian.
/// Returns (x_min, y_min, x_max, y_max) in pixel space.
pub fn gaussian_aabb_2d(
    mean_2d: (f32, f32),
    sigma_2d: &Matrix3<f32>,
    n_sigma: f32,       // typically 3.0 (covers 99.7% of Gaussian mass)
    width: u32,
    height: u32,
) -> Option<(i32, i32, i32, i32)> {
    // Eigenvalues of 2×2 covariance sub-block
    let a = sigma_2d[(0, 0)];
    let b = sigma_2d[(0, 1)];
    let c = sigma_2d[(1, 1)];

    let mid = (a + c) * 0.5;
    let disc = ((a - c) * (a - c) * 0.25 + b * b).sqrt();

    let lambda1 = (mid + disc).max(0.0);
    let lambda2 = (mid - disc).max(0.0);

    let r = n_sigma * lambda1.sqrt().max(lambda2.sqrt());

    let (mx, my) = mean_2d;
    let x0 = ((mx - r) as i32).max(0);
    let y0 = ((my - r) as i32).max(0);
    let x1 = ((mx + r) as i32 + 1).min(width as i32);
    let y1 = ((my + r) as i32 + 1).min(height as i32);

    if x0 >= x1 || y0 >= y1 { None } else { Some((x0, y0, x1, y1)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_matrix_handedness() {
        let proj = make_projection(800.0, 800.0, 320.0, 240.0, 640.0, 480.0, 0.01, 100.0);
        // A point on the near plane z=-near should map to NDC z ≈ -1
        let p_near = proj * Vector4::new(0.0, 0.0, -0.01, 1.0);
        let ndc_z = p_near.z / p_near.w;
        assert!((ndc_z + 1.0).abs() < 0.1, "near plane NDC z: {}", ndc_z);
    }

    #[test]
    fn test_frustum_cull() {
        let identity = Matrix4::identity();
        let proj = make_projection(400.0, 400.0, 200.0, 200.0, 400.0, 400.0, 0.1, 100.0);
        let planes = frustum_planes(&(proj * identity));
        // Point directly in front should pass
        assert!(sphere_in_frustum(&[0.0, 0.0, -5.0], 0.1, &planes));
    }
}