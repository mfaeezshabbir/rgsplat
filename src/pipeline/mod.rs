/// pipeline/mod.rs — Main async pipeline orchestrator
///
/// Stages:
///   1. [async] Video → Frames (ffmpeg subprocess)
///   2. [async] Frames → SfM (COLMAP subprocess)
///   3. [sync, rayon] SfM points → Gaussian initialization
///   4. [async loop] Training (render → loss → backward → Adam → densify)
///   5. [async] Save .ply / .splat output
///
/// Non-blocking: each heavy stage runs in tokio::task::spawn_blocking
/// so the async executor never stalls.

pub mod sfm;
pub mod optimizer;

use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::task;
use anyhow::{Context, Result};
use tracing::{info, warn};
use indicatif::{ProgressBar, ProgressStyle};

use crate::cpu::backward::{self, LossConfig};
use crate::io::{TargetImage, load_images_parallel, save_ply, save_splat};
use crate::math::{Camera, Gaussian3D};
use optimizer::{AdamState, TrainConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline configuration
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub input_video:   PathBuf,
    pub output_dir:    PathBuf,
    pub extract_fps:   f32,
    pub use_gpu:       bool,
    pub train:         TrainConfig,
    pub save_interval: usize,   // save checkpoint every N iters
    pub bg_color:      [f32; 3],
    /// Skip frame extraction + COLMAP and reuse an existing workspace under `output_dir`.
    pub skip_sfm:      bool,
}

impl PipelineConfig {
    pub fn from_video(video: impl Into<PathBuf>, output: impl Into<PathBuf>) -> Self {
        Self {
            input_video: video.into(),
            output_dir: output.into(),
            extract_fps: 2.0,   // 2 fps → good coverage without redundancy
            use_gpu: false,
            train: TrainConfig::default(),
            save_interval: 5000,
            bg_color: [0.0, 0.0, 0.0],
            skip_sfm: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main pipeline entry point
// ─────────────────────────────────────────────────────────────────────────────

pub struct Pipeline {
    cfg: PipelineConfig,
    gpu_ctx: Option<crate::gpu::GpuContext>,
}

impl Pipeline {
    pub async fn new(cfg: PipelineConfig) -> Result<Self> {
        // Attempt GPU initialization
        let gpu_ctx = if cfg.use_gpu {
            match crate::gpu::GpuContext::new().await {
                Ok(ctx) => {
                    info!("GPU ready: {}", ctx.name());
                    Some(ctx)
                }
                Err(e) => {
                    warn!("GPU init failed ({}), falling back to CPU", e);
                    None
                }
            }
        } else {
            info!("GPU disabled, using CPU");
            None
        };

        Ok(Self { cfg, gpu_ctx })
    }

    /// Run the full pipeline end-to-end.
    pub async fn run(&self) -> Result<PathBuf> {
        let t_total = Instant::now();

        let frames_dir = self.cfg.output_dir.join("frames");
        let workspace = self.cfg.output_dir.join("colmap");

        // If the input is a directory, treat it as pre-extracted frames.
        let image_dir = if self.cfg.input_video.is_dir() {
            self.cfg.input_video.clone()
        } else {
            frames_dir.clone()
        };

        if self.cfg.skip_sfm {
            info!("Stages 1-2: skipped (reusing COLMAP workspace at {:?})", workspace);
        } else {
            // ── Stage 1: Frame extraction ─────────────────────────────────────
            if !self.cfg.input_video.is_dir() {
                info!("Stage 1: Extracting frames...");
                let frames =
                    sfm::extract_frames(&self.cfg.input_video, &frames_dir, self.cfg.extract_fps)
                        .await?;
                if frames.is_empty() {
                    anyhow::bail!("No frames extracted from video");
                }
                info!(
                    "Extracted {} frames in {:.1}s",
                    frames.len(),
                    t_total.elapsed().as_secs_f32()
                );
            } else {
                info!("Stage 1: using existing frames in {:?}", image_dir);
            }

            // ── Stage 2: COLMAP SfM ───────────────────────────────────────────
            info!("Stage 2: Running COLMAP SfM (this takes a few minutes)...");
            let t2 = Instant::now();
            sfm::run_colmap(&image_dir, &workspace, self.gpu_ctx.is_some()).await?;
            info!("COLMAP done in {:.1}s", t2.elapsed().as_secs_f32());
        }

        // ── Stage 3: Load SfM output ──────────────────────────────────────
        info!("Stage 3: Parsing SfM output...");
        let (gaussians, cameras) = task::spawn_blocking({
            let workspace = workspace.clone();
            let image_dir = image_dir.clone();
            move || load_sfm_output(&workspace, &image_dir)
        })
        .await??;
        if cameras.is_empty() {
            anyhow::bail!("COLMAP produced no registered cameras");
        }
        info!("Loaded {} Gaussians, {} cameras", gaussians.len(), cameras.len());

        // ── Stage 4: Training ─────────────────────────────────────────────
        info!("Stage 4: Training ({} iterations)...", self.cfg.train.num_iterations);
        let t4 = Instant::now();
        let gaussians = self.train(gaussians, cameras).await?;
        info!("Training done in {:.1}s", t4.elapsed().as_secs_f32());

        // ── Stage 5: Save output ──────────────────────────────────────────
        let output_path = self.cfg.output_dir.join("output.ply");
        let splat_path = self.cfg.output_dir.join("output.splat");
        info!("Stage 5: Saving {} Gaussians to {:?}", gaussians.len(), output_path);
        task::spawn_blocking({
            let ply = output_path.clone();
            let splat = splat_path.clone();
            let g = gaussians.clone();
            move || -> Result<()> {
                save_ply(&g, &ply)?;
                save_splat(&g, &splat)?;
                Ok(())
            }
        })
        .await??;

        info!(
            "Pipeline complete in {:.1}s. Output: {:?}",
            t_total.elapsed().as_secs_f32(),
            output_path
        );
        Ok(output_path)
    }

    // ── Training loop ─────────────────────────────────────────────────────────

    /// Preload target images, then run the (CPU, Rayon-parallel) optimization
    /// loop on a blocking thread so it never stalls the async executor.
    async fn train(
        &self,
        gaussians: Vec<Gaussian3D>,
        cameras: Vec<Camera>,
    ) -> Result<Vec<Gaussian3D>> {
        let images = load_images_parallel(&cameras).await?;
        let cfg = self.cfg.train.clone();
        let bg = self.cfg.bg_color;
        let save_interval = self.cfg.save_interval.max(1);
        let out_dir = self.cfg.output_dir.clone();

        let result = task::spawn_blocking(move || {
            train_blocking(gaussians, &cameras, &images, &cfg, bg, save_interval, &out_dir)
        })
        .await??;
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synchronous training loop (runs inside spawn_blocking)
// ─────────────────────────────────────────────────────────────────────────────

fn train_blocking(
    mut gaussians: Vec<Gaussian3D>,
    cameras: &[Camera],
    images: &[TargetImage],
    cfg: &TrainConfig,
    bg: [f32; 3],
    save_interval: usize,
    out_dir: &Path,
) -> Result<Vec<Gaussian3D>> {
    let mut adam = AdamState::new(gaussians.len());
    let loss_cfg = LossConfig::default();
    let scene_extent = estimate_scene_extent(&gaussians);
    let n_cams = cameras.len();

    let pb = ProgressBar::new(cfg.num_iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")?,
    );

    // Deterministic xorshift for camera shuffling (no external RNG dependency).
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next_cam = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as usize) % n_cams
    };

    for iter in 0..cfg.num_iterations {
        let cam_idx = next_cam();
        let (loss, grads) =
            backward::render_and_backward(&gaussians, &cameras[cam_idx], &images[cam_idx], bg, &loss_cfg);

        optimizer::adam_step(&mut gaussians, &grads, &mut adam, cfg);

        // Densification / pruning.
        if iter >= cfg.densify_from_iter && iter <= cfg.densify_until_iter && iter % 100 == 0 && iter > 0 {
            optimizer::densify_and_prune(&mut gaussians, &mut adam, cfg, scene_extent);
        }

        // Periodic opacity reset.
        if iter % cfg.opacity_reset_iter == 0 && iter > 0 {
            optimizer::reset_opacities(&mut gaussians);
        }

        // Checkpoint.
        if iter % save_interval == 0 && iter > 0 {
            let ckpt = out_dir.join(format!("ckpt_{iter:06}.ply"));
            if let Err(e) = save_ply(&gaussians, &ckpt) {
                warn!("Checkpoint save failed: {e}");
            }
        }

        if iter % 10 == 0 {
            pb.set_message(format!("loss={loss:.4} N={}", gaussians.len()));
        }
        pb.inc(1);
    }

    pb.finish_with_message("Training complete");
    Ok(gaussians)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn load_sfm_output(workspace: &Path, image_dir: &Path) -> Result<(Vec<Gaussian3D>, Vec<Camera>)> {
    use std::fs;

    // Find a COLMAP model directory (sparse/0, sparse/1, ...).
    let sparse_dir = workspace.join("sparse");
    let model_dir = fs::read_dir(&sparse_dir)
        .with_context(|| format!("reading {sparse_dir:?}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .ok_or_else(|| anyhow::anyhow!("No COLMAP model found in {:?}", sparse_dir))?;

    // Points → initial Gaussians.
    let pts_data = fs::read(model_dir.join("points3D.bin"))?;
    let sfm_points = sfm::parse_points3d_bin(&pts_data)?;
    let gaussians = sfm::gaussians_from_sfm(&sfm_points);

    // Intrinsics (keyed by camera_id) + per-image extrinsics → posed cameras.
    let cam_data = fs::read(model_dir.join("cameras.bin"))?;
    let intrinsics = sfm::parse_cameras_bin(&cam_data, 1920, 1080)?;

    let img_data = fs::read(model_dir.join("images.bin"))?;
    let images = sfm::parse_images_bin(&img_data)?;
    let cameras = sfm::build_cameras(&images, &intrinsics, image_dir);

    Ok((gaussians, cameras))
}

fn estimate_scene_extent(gaussians: &[Gaussian3D]) -> f32 {
    if gaussians.is_empty() { return 1.0; }

    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    for g in gaussians {
        for i in 0..3 {
            min[i] = min[i].min(g.position[i]);
            max[i] = max[i].max(g.position[i]);
        }
    }
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}