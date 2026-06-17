/// math/sh.rs — Real Spherical Harmonics (degree 0–3)
///
/// Used to compute view-dependent color: c(d) = Σ_{l=0}^{3} Σ_{m=-l}^{l} k_{lm} Y_{lm}(d)
///
/// Constants from: https://wiki.blender.org/wiki/Reference/Release_Notes/2.6/Render/SphericalHarmonics
/// Coefficients match the 3DGS paper convention.

/// Degree-0 (1 band, 1 coeff)
pub const SH_C0: f32 = 0.28209479177387814; // 1 / (2√π)

/// Degree-1 (3 coeffs)
pub const SH_C1: f32 = 0.4886025119029199; // √(3/(4π))

/// Degree-2 (5 coeffs)
pub const SH_C2: [f32; 5] = [
    1.0925484305920792,   //  ½√(15/π)
   -1.0925484305920792,   // -½√(15/π)
    0.31539156525252005,  //  ¼√(5/π)
   -1.0925484305920792,   // -½√(15/π)
    0.5462742152960396,   //  ¼√(15/π)
];

/// Degree-3 (7 coeffs)
pub const SH_C3: [f32; 7] = [
   -0.5900435899266435,   // -¼√(35/(2π))
    2.890611442640554,    //  ½√(105/π)
   -0.4570457994644658,   // -¼√(21/(2π))
    0.3731763325901154,   //  ¼√(7/π)
   -0.4570457994644658,   // -¼√(21/(2π))
    1.445305721320277,    //  ¼√(105/π)
   -0.5900435899266435,   // -¼√(35/(2π))
];

/// The 16 real-SH basis values (degree 0–3) for a normalized direction.
///
/// `color_ch = 0.5 + Σ_k basis[k] · coeff[ch·16 + k]`. Exposed separately so the
/// forward evaluator and the analytic backward pass share one definition.
#[inline]
pub fn sh_basis(dir: [f32; 3]) -> [f32; 16] {
    let [x, y, z] = dir;
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    [
        SH_C0,
        SH_C1 * -y,
        SH_C1 * z,
        SH_C1 * -x,
        SH_C2[0] * xy,
        SH_C2[1] * yz,
        SH_C2[2] * (2.0 * zz - xx - yy),
        SH_C2[3] * xz,
        SH_C2[4] * (xx - yy),
        SH_C3[0] * y * (3.0 * xx - yy),
        SH_C3[1] * xy * z,
        SH_C3[2] * y * (4.0 * zz - xx - yy),
        SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        SH_C3[4] * x * (4.0 * zz - xx - yy),
        SH_C3[5] * z * (xx - yy),
        SH_C3[6] * x * (xx - 3.0 * yy),
    ]
}

/// Gradient of each SH basis value w.r.t. the direction components `(x, y, z)`.
/// Returns `[∂basis_k/∂x, ∂basis_k/∂y, ∂basis_k/∂z]` for `k = 0..16`.
///
/// Used to backpropagate view-dependent color into the Gaussian position (the
/// direction depends on where the Gaussian sits relative to the camera).
pub fn sh_basis_grad(dir: [f32; 3]) -> [[f32; 3]; 16] {
    let [x, y, z] = dir;
    let (xx, yy, zz) = (x * x, y * y, z * z);
    [
        [0.0, 0.0, 0.0],
        [0.0, -SH_C1, 0.0],
        [0.0, 0.0, SH_C1],
        [-SH_C1, 0.0, 0.0],
        [SH_C2[0] * y, SH_C2[0] * x, 0.0],
        [0.0, SH_C2[1] * z, SH_C2[1] * y],
        [SH_C2[2] * -2.0 * x, SH_C2[2] * -2.0 * y, SH_C2[2] * 4.0 * z],
        [SH_C2[3] * z, 0.0, SH_C2[3] * x],
        [SH_C2[4] * 2.0 * x, SH_C2[4] * -2.0 * y, 0.0],
        [SH_C3[0] * 6.0 * x * y, SH_C3[0] * 3.0 * (xx - yy), 0.0],
        [SH_C3[1] * y * z, SH_C3[1] * x * z, SH_C3[1] * x * y],
        [SH_C3[2] * -2.0 * x * y, SH_C3[2] * (4.0 * zz - xx - 3.0 * yy), SH_C3[2] * 8.0 * y * z],
        [SH_C3[3] * -6.0 * x * z, SH_C3[3] * -6.0 * y * z, SH_C3[3] * (6.0 * zz - 3.0 * xx - 3.0 * yy)],
        [SH_C3[4] * (4.0 * zz - 3.0 * xx - yy), SH_C3[4] * -2.0 * x * y, SH_C3[4] * 8.0 * x * z],
        [SH_C3[5] * 2.0 * x * z, SH_C3[5] * -2.0 * y * z, SH_C3[5] * (xx - yy)],
        [SH_C3[6] * 3.0 * (xx - yy), SH_C3[6] * -6.0 * x * y, 0.0],
    ]
}

/// Raw (pre-clamp) SH color: `0.5 + Σ_k basis[k]·coeff[ch·16+k]` per channel.
#[inline]
pub fn eval_sh_raw(coeffs: &[f32; 48], basis: &[f32; 16]) -> [f32; 3] {
    let mut result = [0.5f32; 3];
    for ch in 0..3 {
        let c = &coeffs[ch * 16..(ch + 1) * 16];
        let mut acc = 0.0f32;
        for k in 0..16 {
            acc += basis[k] * c[k];
        }
        result[ch] += acc;
    }
    result
}

/// Evaluate spherical harmonics up to degree 3 for a view direction.
///
/// `coeffs` is stored as `[16 R-coeffs, 16 G-coeffs, 16 B-coeffs]`; `dir` is the
/// normalized view direction (camera − Gaussian). Returns RGB clamped to [0, 1].
pub fn eval_sh_deg3(coeffs: &[f32; 48], dir: [f32; 3]) -> [f32; 3] {
    let basis = sh_basis(dir);
    let raw = eval_sh_raw(coeffs, &basis);
    [
        raw[0].clamp(0.0, 1.0),
        raw[1].clamp(0.0, 1.0),
        raw[2].clamp(0.0, 1.0),
    ]
}

/// Normalize a direction vector. Returns [0,0,1] if near-zero.
#[inline]
pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-8 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// View direction from Gaussian center to camera
#[inline]
pub fn view_dir(gaussian_pos: [f32; 3], camera_pos: [f32; 3]) -> [f32; 3] {
    normalize([
        camera_pos[0] - gaussian_pos[0],
        camera_pos[1] - gaussian_pos[1],
        camera_pos[2] - gaussian_pos[2],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sh_dc_only() {
        // With only DC coefficient (l=0), color should be 0.5 + SH_C0 * coeff
        let mut coeffs = [0.0f32; 48];
        // R channel DC
        coeffs[0] = 1.0 / SH_C0;  // so result = 0.5 + SH_C0 * (1/SH_C0) = 1.5 → clamp to 1.0
        let dir = [0.0, 0.0, 1.0];
        let c = eval_sh_deg3(&coeffs, dir);
        assert!((c[0] - 1.0).abs() < 1e-5, "R should clamp to 1.0, got {}", c[0]);
        assert!((c[1] - 0.5).abs() < 1e-5, "G should be 0.5, got {}", c[1]);
        assert!((c[2] - 0.5).abs() < 1e-5, "B should be 0.5, got {}", c[2]);
    }
}