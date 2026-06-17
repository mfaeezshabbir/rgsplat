//! GPU smoke test (only built with `--features gpu`). Validates that the WGSL
//! shader compiles and the projection pass runs. Skips gracefully when no GPU
//! adapter is available (e.g. headless CI).

#![cfg(feature = "gpu")]

use rgsplat::gpu::GpuRenderer;
use rgsplat::math::{Camera, Gaussian3D, logit};

fn cam(w: u32, h: u32) -> Camera {
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

#[test]
fn gpu_projection_smoke() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let renderer = match GpuRenderer::new().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no GPU available, skipping: {e}");
                return;
            }
        };
        eprintln!("GPU adapter: {}", renderer.adapter_name());

        let camera = cam(32, 32);
        let mut g = Gaussian3D::new([0.0, 0.0, 2.0]);
        g.opacity_logit = logit(0.9);
        g.log_scale = [-1.5, -1.5, -1.5];
        g.sh_coeffs[0] = 1.0;

        let splats = renderer.project(&[g], &camera).await.unwrap();
        assert_eq!(splats.len(), 1);
        let s = &splats[0];
        assert!(s.alpha > 0.0 && s.depth > 0.0, "splat should be visible: {s:?}");
        assert!(s.sx.is_finite() && s.sy.is_finite());

        // Full render should produce a lit center pixel.
        let img = renderer.render(&[g], &camera, [0.0, 0.0, 0.0]).await.unwrap();
        let c = (16 * 32 + 16) * 3;
        assert!(img[c] > 0.0, "center pixel should be lit");
    });
}
