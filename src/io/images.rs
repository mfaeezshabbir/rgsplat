//! io/images.rs — load training target images as linear `f32` RGBA buffers.
//!
//! The CPU rasterizer renders in **linear** light and only gamma-encodes on
//! display, so targets are decoded sRGB → linear here to keep the photometric
//! loss in a consistent space. Each target is resized to its camera's
//! resolution if the file dimensions differ.

use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::math::Camera;

/// A decoded training image: row-major linear RGBA, `width * height * 4` floats.
#[derive(Clone)]
pub struct TargetImage {
    pub width: usize,
    pub height: usize,
    /// Linear RGBA in `[0, 1]`, pixel `(x, y)` at `(y * width + x) * 4`.
    pub data: Vec<f32>,
}

impl TargetImage {
    /// A solid black opaque image (used when a camera has no associated file).
    pub fn black(width: usize, height: usize) -> Self {
        let mut data = vec![0.0f32; width * height * 4];
        for px in data.chunks_exact_mut(4) {
            px[3] = 1.0;
        }
        Self { width, height, data }
    }
}

/// Decode every camera's image in parallel. Cameras without an `image_path`
/// yield a black image of the camera's size.
pub async fn load_images_parallel(cameras: &[Camera]) -> Result<Vec<TargetImage>> {
    let jobs: Vec<(Option<std::path::PathBuf>, usize, usize)> = cameras
        .iter()
        .map(|c| (c.image_path.clone(), c.width as usize, c.height as usize))
        .collect();

    tokio::task::spawn_blocking(move || {
        jobs.par_iter()
            .map(|(path, w, h)| match path {
                Some(p) => load_target(p, *w, *h),
                None => Ok(TargetImage::black(*w, *h)),
            })
            .collect::<Result<Vec<_>>>()
    })
    .await?
}

/// Load a single image, convert to linear RGBA, and resize to `(w, h)`.
pub fn load_target(path: &Path, w: usize, h: usize) -> Result<TargetImage> {
    let img = image::open(path).with_context(|| format!("decoding image {path:?}"))?;
    let img = if img.width() as usize != w || img.height() as usize != h {
        img.resize_exact(w as u32, h as u32, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgba = img.to_rgba8();

    let mut data = vec![0.0f32; w * h * 4];
    for (dst, src) in data.chunks_exact_mut(4).zip(rgba.pixels()) {
        dst[0] = srgb_to_linear(src[0] as f32 / 255.0);
        dst[1] = srgb_to_linear(src[1] as f32 / 255.0);
        dst[2] = srgb_to_linear(src[2] as f32 / 255.0);
        dst[3] = src[3] as f32 / 255.0; // alpha is linear
    }
    Ok(TargetImage { width: w, height: h, data })
}

/// sRGB → linear (inverse of the rasterizer's `linear_to_srgb`).
#[inline]
pub fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.040_45 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_endpoints() {
        assert!((srgb_to_linear(0.0)).abs() < 1e-6);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-5);
        // sRGB is darker than linear in the midtones.
        assert!(srgb_to_linear(0.5) < 0.5);
    }

    #[test]
    fn black_image_has_opaque_alpha() {
        let img = TargetImage::black(4, 4);
        assert_eq!(img.data.len(), 4 * 4 * 4);
        assert_eq!(img.data[3], 1.0);
    }
}
