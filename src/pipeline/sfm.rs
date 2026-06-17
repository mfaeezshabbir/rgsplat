/// pipeline/sfm.rs — Structure from Motion initialization
///
/// Extracts frames from video, runs COLMAP (external) for camera pose estimation,
/// then loads the sparse point cloud as Gaussian seed positions.
///
/// Pipeline:
///   Video → Frames → COLMAP → sparse/points3D.bin → Vec<Gaussian3D>
///
/// This module handles:
/// 1. Frame extraction (async, memory-mapped)
/// 2. COLMAP binary file parsing (points3D.bin, cameras.bin, images.bin)
/// 3. Initial Gaussian creation from SfM points

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::process::Command;
use tokio::fs;
use anyhow::{Result, Context, bail};
use tracing::{info, warn};

use crate::math::{Gaussian3D, Camera};

// ─────────────────────────────────────────────────────────────────────────────
// Frame extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract frames from a video file using ffmpeg.
///
/// Non-blocking: spawns ffmpeg as a subprocess, doesn't occupy thread pool.
/// Frame rate reduced to `fps` to save disk I/O and COLMAP compute.
pub async fn extract_frames(
    video_path: &Path,
    output_dir: &Path,
    fps: f32,
) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(output_dir).await?;

    let frame_pattern = output_dir.join("frame_%06d.png");
    info!("Extracting frames at {}fps from {:?}", fps, video_path);

    let status = Command::new("ffmpeg")
        .args([
            "-i", video_path.to_str().unwrap(),
            "-vf", &format!("fps={}", fps),
            "-qscale:v", "1",          // highest quality
            "-f", "image2",
            frame_pattern.to_str().unwrap(),
            "-y",                       // overwrite
            "-loglevel", "error",
        ])
        .status()
        .await
        .context("Failed to spawn ffmpeg. Is ffmpeg installed?")?;

    if !status.success() {
        bail!("ffmpeg exited with error: {:?}", status);
    }

    // Collect extracted frame paths
    let mut frames = Vec::new();
    let mut entries = fs::read_dir(output_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("png") {
            frames.push(path);
        }
    }
    frames.sort();
    info!("Extracted {} frames", frames.len());
    Ok(frames)
}

/// Run COLMAP automatic reconstruction.
/// 
/// Requires COLMAP installed. Uses feature extraction → matching → mapping pipeline.
pub async fn run_colmap(
    image_dir: &Path,
    workspace_dir: &Path,
    use_gpu: bool,
) -> Result<()> {
    let db_path = workspace_dir.join("colmap.db");
    let sparse_dir = workspace_dir.join("sparse");
    fs::create_dir_all(&sparse_dir).await?;

    let gpu_flag = if use_gpu { "1" } else { "0" };
    info!("Running COLMAP (GPU={})...", use_gpu);

    // Step 1: Feature extraction (SIFT)
    run_colmap_cmd(&[
        "feature_extractor",
        "--database_path", db_path.to_str().unwrap(),
        "--image_path", image_dir.to_str().unwrap(),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu_flag,
    ]).await.context("COLMAP feature extraction failed")?;

    // Step 2: Exhaustive matching
    run_colmap_cmd(&[
        "exhaustive_matcher",
        "--database_path", db_path.to_str().unwrap(),
        "--SiftMatching.use_gpu", gpu_flag,
    ]).await.context("COLMAP matching failed")?;

    // Step 3: Sparse reconstruction (incremental SfM)
    run_colmap_cmd(&[
        "mapper",
        "--database_path", db_path.to_str().unwrap(),
        "--image_path", image_dir.to_str().unwrap(),
        "--output_path", sparse_dir.to_str().unwrap(),
    ]).await.context("COLMAP mapper failed")?;

    info!("COLMAP complete. Sparse model at {:?}", sparse_dir);
    Ok(())
}

async fn run_colmap_cmd(args: &[&str]) -> Result<()> {
    let status = Command::new("colmap")
        .args(args)
        .status()
        .await
        .context("Failed to spawn COLMAP. Is colmap installed?")?;
    if !status.success() {
        bail!("COLMAP {:?} failed: {:?}", args[0], status);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// COLMAP binary format parsers
// ─────────────────────────────────────────────────────────────────────────────

/// A 3D point from COLMAP sparse reconstruction
#[derive(Debug, Clone)]
pub struct SfMPoint {
    pub id: u64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub error: f64,           // reprojection error
    pub track_length: usize,  // number of observations
}

/// Parse COLMAP points3D.bin
///
/// Binary format per point:
///   u64 point_id, f64 x, f64 y, f64 z, u8 r, g, b, f64 error,
///   u64 track_length, then track_length × (u32 image_id, u32 point2d_idx)
pub fn parse_points3d_bin(data: &[u8]) -> Result<Vec<SfMPoint>> {
    use std::io::{Cursor, Read};

    let mut cursor = Cursor::new(data);
    let mut points = Vec::new();

    // Read u64 count header
    let num_pts = read_u64(&mut cursor)?;
    points.reserve(num_pts as usize);

    for _ in 0..num_pts {
        let id = read_u64(&mut cursor)?;
        let x = read_f64(&mut cursor)?;
        let y = read_f64(&mut cursor)?;
        let z = read_f64(&mut cursor)?;
        let r = read_u8(&mut cursor)?;
        let g = read_u8(&mut cursor)?;
        let b = read_u8(&mut cursor)?;
        let error = read_f64(&mut cursor)?;
        let track_len = read_u64(&mut cursor)? as usize;

        // Skip track data (image_id + point2d_idx pairs)
        let skip = track_len * 8;
        let mut skip_buf = vec![0u8; skip];
        cursor.read_exact(&mut skip_buf)?;

        points.push(SfMPoint {
            id,
            xyz: [x, y, z],
            rgb: [r, g, b],
            error,
            track_length: track_len,
        });
    }

    info!("Parsed {} SfM points", points.len());
    Ok(points)
}

/// Parse COLMAP cameras.bin
/// Returns map: camera_id → Camera
pub fn parse_cameras_bin(data: &[u8], width: u32, height: u32) -> Result<HashMap<u32, Camera>> {
    use std::io::Cursor;

    let mut cursor = Cursor::new(data);
    let num_cams = read_u64(&mut cursor)?;
    let mut cameras = HashMap::new();

    for _ in 0..num_cams {
        let cam_id = read_u32(&mut cursor)?;
        let model_id = read_i32(&mut cursor)?;  // 0=SIMPLE_PINHOLE, 1=PINHOLE, ...
        let w = read_u64(&mut cursor)? as u32;
        let h = read_u64(&mut cursor)? as u32;
        let num_params = colmap_camera_params(model_id);
        let mut params = vec![0.0f64; num_params];
        for p in &mut params {
            *p = read_f64(&mut cursor)?;
        }
        // Use the camera's actual resolution; fall back to the caller's hint.
        let (cw, ch) = if w > 0 && h > 0 { (w, h) } else { (width, height) };

        // Intrinsics depend on the model: PINHOLE = [fx, fy, cx, cy];
        // SIMPLE_* and *RADIAL = [f, cx, cy, ...].
        let (fx, fy, cx, cy) = match model_id {
            1 | 4 if params.len() >= 4 => {
                (params[0] as f32, params[1] as f32, params[2] as f32, params[3] as f32)
            }
            _ if params.len() >= 3 => {
                (params[0] as f32, params[0] as f32, params[1] as f32, params[2] as f32)
            }
            _ => (cw as f32, ch as f32, cw as f32 / 2.0, ch as f32 / 2.0),
        };

        cameras.insert(cam_id, Camera {
            width: cw, height: ch, fx, fy, cx, cy,
            c2w: [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
            distortion: [0.0; 3],
            image_path: None,
        });
    }

    Ok(cameras)
}

// ─────────────────────────────────────────────────────────────────────────────
// COLMAP images.bin — per-image extrinsics (the camera poses)
// ─────────────────────────────────────────────────────────────────────────────

/// One registered image: world→camera rotation `qvec` (Hamilton, `[w,x,y,z]`),
/// translation `tvec`, the `camera_id` into the intrinsics map, and the file name.
#[derive(Debug, Clone)]
pub struct ColmapImage {
    pub image_id: u32,
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: u32,
    pub name: String,
}

/// Parse COLMAP `images.bin`.
///
/// Layout: `u64` registered-image count, then per image:
/// `u32 image_id`, `4×f64 qvec(w,x,y,z)`, `3×f64 tvec`, `u32 camera_id`,
/// NUL-terminated `name`, `u64 num_points2D`, then `num_points2D ×
/// (f64 x, f64 y, i64 point3d_id)` (we skip the 2D observations).
pub fn parse_images_bin(data: &[u8]) -> Result<Vec<ColmapImage>> {
    use std::io::Read;

    let mut cursor = std::io::Cursor::new(data);
    let num_images = read_u64(&mut cursor)?;
    let mut images = Vec::with_capacity(num_images as usize);

    for _ in 0..num_images {
        let image_id = read_u32(&mut cursor)?;
        let qvec = [
            read_f64(&mut cursor)?,
            read_f64(&mut cursor)?,
            read_f64(&mut cursor)?,
            read_f64(&mut cursor)?,
        ];
        let tvec = [
            read_f64(&mut cursor)?,
            read_f64(&mut cursor)?,
            read_f64(&mut cursor)?,
        ];
        let camera_id = read_u32(&mut cursor)?;

        // NUL-terminated image name.
        let mut name_bytes = Vec::new();
        loop {
            let b = read_u8(&mut cursor)?;
            if b == 0 {
                break;
            }
            name_bytes.push(b);
        }
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        // Skip the 2D point observations: each is 8 + 8 + 8 bytes.
        let num_pts2d = read_u64(&mut cursor)?;
        let skip = num_pts2d as usize * 24;
        let mut skip_buf = vec![0u8; skip];
        cursor.read_exact(&mut skip_buf)?;

        images.push(ColmapImage { image_id, qvec, tvec, camera_id, name });
    }

    info!("Parsed {} camera poses from images.bin", images.len());
    Ok(images)
}

/// Build training [`Camera`]s by joining per-image extrinsics with their
/// intrinsics and resolving each image file under `image_dir`.
///
/// COLMAP stores the world→camera pose `(R(q), t)`, so
/// `c2w = [[Rᵀ | −Rᵀt], [0 0 0 1]]` (Rᵀ rotation, camera center −Rᵀt).
pub fn build_cameras(
    images: &[ColmapImage],
    intrinsics: &HashMap<u32, Camera>,
    image_dir: &Path,
) -> Vec<Camera> {
    use crate::math::quaternion;

    let mut cameras = Vec::with_capacity(images.len());
    for img in images {
        let Some(intr) = intrinsics.get(&img.camera_id) else {
            warn!("image {} references unknown camera {}", img.name, img.camera_id);
            continue;
        };

        let q = [img.qvec[0] as f32, img.qvec[1] as f32, img.qvec[2] as f32, img.qvec[3] as f32];
        let r = quaternion::to_rotation_matrix(q); // world→camera rotation
        let rt = r.transpose(); // camera→world rotation
        let t = nalgebra::Vector3::new(img.tvec[0] as f32, img.tvec[1] as f32, img.tvec[2] as f32);
        let center = -(rt * t); // camera center in world space

        let mut c2w = [[0.0f32; 4]; 4];
        for row in 0..3 {
            for col in 0..3 {
                c2w[row][col] = rt[(row, col)];
            }
            c2w[row][3] = center[row];
        }
        c2w[3] = [0.0, 0.0, 0.0, 1.0];

        let mut cam = intr.clone();
        cam.c2w = c2w;
        cam.image_path = Some(image_dir.join(&img.name));
        cameras.push(cam);
    }

    info!("Built {} posed cameras", cameras.len());
    cameras
}

fn colmap_camera_params(model_id: i32) -> usize {
    match model_id {
        0 => 3,   // SIMPLE_PINHOLE: f, cx, cy
        1 => 4,   // PINHOLE: fx, fy, cx, cy
        2 => 4,   // SIMPLE_RADIAL: f, cx, cy, k1
        3 => 5,   // RADIAL: f, cx, cy, k1, k2
        4 => 8,   // OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        _ => 4,   // default
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialize Gaussians from SfM points
// ─────────────────────────────────────────────────────────────────────────────

/// Convert SfM point cloud to initial Gaussians.
///
/// Initial parameters:
/// - Position: from SfM
/// - Color: from SfM RGB (encoded as SH DC coefficient)
/// - Scale: based on mean nearest-neighbor distance
/// - Rotation: identity
/// - Opacity: 0.1 (will be optimized)
pub fn gaussians_from_sfm(points: &[SfMPoint]) -> Vec<Gaussian3D> {
    if points.is_empty() {
        warn!("No SfM points to initialize Gaussians from");
        return Vec::new();
    }

    // Estimate initial scale: average distance to 3 nearest neighbors
    // Approximated with a spatial grid for O(n log n) vs O(n²)
    let scales = estimate_initial_scales(points);

    let mut gaussians = Vec::with_capacity(points.len());
    const SH_C0_INV: f32 = 1.0 / 0.28209479177387814;

    for (i, pt) in points.iter().enumerate() {
        let mut g = Gaussian3D::new([
            pt.xyz[0] as f32,
            pt.xyz[1] as f32,
            pt.xyz[2] as f32,
        ]);

        // Encode RGB as SH DC (degree 0)
        // color = 0.5 + SH_C0 * dc  →  dc = (color - 0.5) * SH_C0_INV
        let r = pt.rgb[0] as f32 / 255.0;
        let gv = pt.rgb[1] as f32 / 255.0;
        let b = pt.rgb[2] as f32 / 255.0;
        // sRGB → linear approximation
        let to_lin = |x: f32| x.powf(2.2);
        g.sh_coeffs[0]  = (to_lin(r) - 0.5) * SH_C0_INV;  // R DC
        g.sh_coeffs[16] = (to_lin(gv) - 0.5) * SH_C0_INV; // G DC
        g.sh_coeffs[32] = (to_lin(b) - 0.5) * SH_C0_INV;  // B DC

        // Set initial scale (isotropic)
        let s = scales[i].max(1e-4).ln();
        g.log_scale = [s, s, s];

        // Low initial opacity
        g.opacity_logit = crate::math::logit(0.1);

        gaussians.push(g);
    }

    info!("Initialized {} Gaussians from SfM", gaussians.len());
    gaussians
}

/// Estimate per-point initial scale as mean distance to K nearest neighbors.
/// Uses a simple spatial sorting approach for efficiency.
fn estimate_initial_scales(points: &[SfMPoint]) -> Vec<f32> {
    use rayon::prelude::*;

    const K: usize = 3;

    // Sort points by x for approximate nearest-neighbor search
    let mut sorted: Vec<(usize, [f64; 3])> = points.iter()
        .enumerate()
        .map(|(i, p)| (i, p.xyz))
        .collect();
    sorted.sort_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap());

    let mut scales = vec![0.1f32; points.len()];

    // For each point, search within a window for KNN
    scales.par_iter_mut().enumerate().for_each(|(i, scale)| {
        let xyz = points[i].xyz;
        let window = 50.min(points.len());
        let start = sorted.binary_search_by(|s| s.1[0].partial_cmp(&xyz[0]).unwrap())
            .unwrap_or(0)
            .saturating_sub(window / 2);
        let end = (start + window).min(sorted.len());

        let mut dists: Vec<f64> = sorted[start..end].iter()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| {
                let dx = p[0] - xyz[0];
                let dy = p[1] - xyz[1];
                let dz = p[2] - xyz[2];
                (dx*dx + dy*dy + dz*dz).sqrt()
            })
            .collect();

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if dists.is_empty() {
            *scale = 0.01;
        } else {
            let k = K.min(dists.len());
            *scale = (dists[..k].iter().sum::<f64>() / k as f64) as f32;
        }
    });

    scales
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary reader helpers (little-endian)
// ─────────────────────────────────────────────────────────────────────────────

fn read_u8(c: &mut std::io::Cursor<&[u8]>) -> Result<u8> {
    use std::io::Read;
    let mut buf = [0u8; 1];
    c.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32(c: &mut std::io::Cursor<&[u8]>) -> Result<u32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    c.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(c: &mut std::io::Cursor<&[u8]>) -> Result<i32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    c.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(c: &mut std::io::Cursor<&[u8]>) -> Result<u64> {
    use std::io::Read;
    let mut buf = [0u8; 8];
    c.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(c: &mut std::io::Cursor<&[u8]>) -> Result<f64> {
    use std::io::Read;
    let mut buf = [0u8; 8];
    c.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}