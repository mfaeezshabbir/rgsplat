//! io/cameras.rs — NeRF-style `transforms.json` camera import/export.
//!
//! This is an alternative to COLMAP for supplying camera intrinsics/extrinsics
//! (e.g. from Instant-NGP, Blender, or Nerfstudio). The `transform_matrix`
//! entries are camera-to-world in the **OpenGL** convention (+x right, +y up,
//! −z forward); we convert to the **OpenCV** convention (+x right, +y down,
//! +z forward) used throughout the rest of the pipeline.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::math::Camera;

#[derive(Debug, Deserialize, Serialize)]
struct Frame {
    file_path: String,
    transform_matrix: [[f32; 4]; 4],
}

#[derive(Debug, Deserialize, Serialize)]
struct Transforms {
    #[serde(default)]
    camera_angle_x: Option<f32>,
    #[serde(default)]
    fl_x: Option<f32>,
    #[serde(default)]
    fl_y: Option<f32>,
    #[serde(default)]
    cx: Option<f32>,
    #[serde(default)]
    cy: Option<f32>,
    #[serde(default)]
    w: Option<f32>,
    #[serde(default)]
    h: Option<f32>,
    frames: Vec<Frame>,
}

/// Load cameras from a `transforms.json`. `base_dir` resolves relative
/// `file_path`s; `default_wh` is used if the JSON omits resolution.
pub fn load_transforms_json(
    path: &Path,
    base_dir: &Path,
    default_wh: (u32, u32),
) -> Result<Vec<Camera>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    let t: Transforms = serde_json::from_str(&text).context("parsing transforms.json")?;

    let width = t.w.map(|v| v as u32).unwrap_or(default_wh.0);
    let height = t.h.map(|v| v as u32).unwrap_or(default_wh.1);

    // Focal length: prefer fl_x/fl_y, else derive from horizontal FOV.
    let fx = t.fl_x.unwrap_or_else(|| match t.camera_angle_x {
        Some(a) => 0.5 * width as f32 / (0.5 * a).tan(),
        None => width as f32, // last-resort default
    });
    let fy = t.fl_y.unwrap_or(fx);
    let cx = t.cx.unwrap_or(width as f32 / 2.0);
    let cy = t.cy.unwrap_or(height as f32 / 2.0);

    let mut cameras = Vec::with_capacity(t.frames.len());
    for f in &t.frames {
        let c2w = opengl_to_opencv_c2w(&f.transform_matrix);
        let image_path = resolve_path(base_dir, &f.file_path);
        cameras.push(Camera {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            c2w,
            distortion: [0.0; 3],
            image_path: Some(image_path),
        });
    }
    Ok(cameras)
}

/// Resolve a frame `file_path` against `base_dir`, tolerating a missing
/// extension (NeRF synthetic data omits `.png`).
fn resolve_path(base_dir: &Path, file_path: &str) -> PathBuf {
    let p = base_dir.join(file_path);
    if p.exists() {
        return p;
    }
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG"] {
        let with_ext = base_dir.join(format!("{file_path}.{ext}"));
        if with_ext.exists() {
            return with_ext;
        }
    }
    p
}

/// Convert an OpenGL camera-to-world matrix to the OpenCV convention by
/// negating the camera-space Y and Z basis vectors (columns 1 and 2).
fn opengl_to_opencv_c2w(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = *m;
    for row in 0..3 {
        out[row][1] = -m[row][1];
        out[row][2] = -m[row][2];
    }
    out
}

/// Write cameras to a `transforms.json` (OpenCV → OpenGL on the way out).
pub fn save_transforms_json(cameras: &[Camera], path: &Path) -> Result<()> {
    let frames = cameras
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut m = c.c2w;
            for row in 0..3 {
                m[row][1] = -c.c2w[row][1];
                m[row][2] = -c.c2w[row][2];
            }
            Frame {
                file_path: c
                    .image_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("frame_{i:06}")),
                transform_matrix: m,
            }
        })
        .collect();

    let first = cameras.first();
    let t = Transforms {
        camera_angle_x: first.map(|c| 2.0 * (0.5 * c.width as f32 / c.fx).atan()),
        fl_x: first.map(|c| c.fx),
        fl_y: first.map(|c| c.fy),
        cx: first.map(|c| c.cx),
        cy: first.map(|c| c.cy),
        w: first.map(|c| c.width as f32),
        h: first.map(|c| c.height as f32),
        frames,
    };

    let text = serde_json::to_string_pretty(&t)?;
    std::fs::write(path, text).with_context(|| format!("writing {path:?}"))?;
    Ok(())
}
