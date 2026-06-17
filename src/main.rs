//! `rgsplat` — command-line driver for the Gaussian Splatting pipeline.
//!
//! ```text
//! rgsplat --input scene.mp4 --output ./out            # CPU
//! rgsplat --input scene.mp4 --output ./out --gpu      # GPU (needs `--features gpu`)
//! rgsplat --input ./images  --output ./out --skip-extract  # pre-extracted frames
//! ```

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};

use rgsplat::pipeline::{Pipeline, PipelineConfig};

#[derive(Parser, Debug)]
#[command(
    name = "rgsplat",
    version,
    about = "Train a 3D Gaussian Splatting model from a video or image set"
)]
struct Cli {
    /// Input video file (e.g. scene.mp4) or a directory of frames.
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for frames, COLMAP workspace, checkpoints and the final PLY.
    #[arg(short, long)]
    output: PathBuf,

    /// Use the GPU compute path if available (requires building with `--features gpu`).
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Frames per second to extract from the input video.
    #[arg(long, default_value_t = 2.0)]
    fps: f32,

    /// Number of training iterations.
    #[arg(long, default_value_t = 30_000)]
    iterations: usize,

    /// Save a checkpoint every N iterations.
    #[arg(long, default_value_t = 5_000)]
    save_interval: usize,

    /// Background color as `R,G,B` floats in [0,1].
    #[arg(long, default_value = "0,0,0", value_parser = parse_rgb)]
    bg: [f32; 3],

    /// Skip frame extraction / SfM and load an existing COLMAP workspace from `--output`.
    #[arg(long, default_value_t = false)]
    skip_sfm: bool,
}

fn parse_rgb(s: &str) -> Result<[f32; 3], String> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    match parts.as_slice() {
        [r, g, b] => Ok([*r, *g, *b]),
        _ => Err("expected exactly 3 comma-separated floats, e.g. 0,0,0".into()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Honor RUST_LOG, default to `info`.
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let mut cfg = PipelineConfig::from_video(&cli.input, &cli.output);
    cfg.extract_fps = cli.fps;
    cfg.use_gpu = cli.gpu;
    cfg.save_interval = cli.save_interval;
    cfg.bg_color = cli.bg;
    cfg.skip_sfm = cli.skip_sfm;
    cfg.train.num_iterations = cli.iterations;

    let pipeline = Pipeline::new(cfg).await?;
    let output = pipeline.run().await?;

    tracing::info!("Done. Trained model written to {}", output.display());
    Ok(())
}
