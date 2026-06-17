//! End-to-end check: the differentiable rasterizer + Adam actually optimize a
//! tiny scene so the photometric loss decreases. This exercises the full
//! render → backward → Adam path the pipeline relies on.

use gaussian_splat_pipeline::cpu::backward::{LossConfig, render_and_backward, render_color};
use gaussian_splat_pipeline::io::TargetImage;
use gaussian_splat_pipeline::math::{Camera, Gaussian3D, logit};
use gaussian_splat_pipeline::pipeline::optimizer::{AdamState, TrainConfig, adam_step};

fn identity_camera(w: u32, h: u32) -> Camera {
    Camera {
        width: w,
        height: h,
        fx: w as f32,
        fy: w as f32,
        cx: w as f32 / 2.0,
        cy: h as f32 / 2.0,
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

fn solid_target(w: usize, h: usize, color: [f32; 3]) -> TargetImage {
    let mut data = vec![0.0f32; w * h * 4];
    for px in data.chunks_exact_mut(4) {
        px[..3].copy_from_slice(&color);
        px[3] = 1.0;
    }
    TargetImage { width: w, height: h, data }
}

#[test]
fn training_reduces_photometric_loss() {
    let (w, h) = (32u32, 32u32);
    let cam = identity_camera(w, h);
    let bg = [0.0, 0.0, 0.0];
    let cfg = LossConfig { lambda_dssim: 0.2 };
    // Target: a uniform teal-ish color the Gaussians must learn to reproduce.
    let target = solid_target(w as usize, h as usize, [0.2, 0.55, 0.5]);

    // A few wide Gaussians blanketing the view, deliberately mis-colored.
    let mut gaussians = Vec::new();
    for &(x, y) in &[(-0.3f32, -0.3f32), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3), (0.0, 0.0)] {
        let mut g = Gaussian3D::new([x, y, 2.0]);
        g.opacity_logit = logit(0.5);
        g.log_scale = [-1.3, -1.3, -1.3];
        // Start grayish (wrong) so there is something to optimize.
        for ch in 0..3 {
            g.sh_coeffs[ch * 16] = 0.0;
        }
        gaussians.push(g);
    }

    let train_cfg = TrainConfig::default();
    let mut adam = AdamState::new(gaussians.len());

    let initial = render_and_backward(&gaussians, &cam, &target, bg, &cfg).0;

    let mut last = initial;
    for _ in 0..200 {
        let (loss, grads) = render_and_backward(&gaussians, &cam, &target, bg, &cfg);
        adam_step(&mut gaussians, &grads, &mut adam, &train_cfg);
        last = loss;
    }

    assert!(
        last < initial * 0.5,
        "expected loss to drop substantially: initial {initial:.4} → final {last:.4}"
    );

    // The rendered center pixel should move toward the target color.
    let img = render_color(&gaussians, &cam, bg);
    let c = ((h as usize / 2) * w as usize + w as usize / 2) * 3;
    for ch in 0..3 {
        let err = (img[c + ch] - target.data[ch]).abs();
        assert!(err < 0.15, "channel {ch} far from target: {} vs {}", img[c + ch], target.data[ch]);
    }
}
