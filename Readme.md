# Gaussian Splatting Pipeline

A high-performance implementation of 3D Gaussian Splatting for real-time radiance field rendering from video, written in Rust with GPU acceleration via wgpu.

## Overview

This crate implements the complete pipeline described in [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., SIGGRAPH 2023). It takes a video as input and produces a trained `.ply` point cloud of 3D Gaussians that can be rendered in real time.

### Features

- **Differentiable CPU rasterizer**: Tile-based forward render plus a fully
  analytic backward pass (gradients for SH color, opacity, position, scale and
  rotation), validated against finite differences. Loss is L1 + D-SSIM.
- **GPU-accelerated projection**: An optional wgpu compute pass projects
  Gaussians on the GPU for fast inference/preview (`--features gpu`); the
  differentiable training path runs on the CPU.
- **Async architecture**: Built with Tokio for non-blocking I/O and subprocess management
- **Parallel processing**: Uses Rayon for data-parallel rendering and optimization
- **Standard formats**: Reads/writes the 3DGS `.ply` interchange format and the
  compact `.splat` runtime format
- **Python bindings**: Available via maturin (`--features python`)

## Documentation

For detailed documentation, including:
- Installation instructions
- Quick start guide
- Pipeline architecture
- API reference
- Shader documentation
- Configuration options
- Performance tuning guides

Please see the [full documentation](./docs/docs.html) in the `docs/` directory.

## Quick Install

One command to install everything (Rust, ffmpeg, COLMAP, and rgsplat):

**Linux / macOS**
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/mfaeezshabbir/rgsplat/main/scripts/install.sh)
```

**Windows (PowerShell as Admin)**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

Or clone and run the scripts locally:
```bash
git clone https://github.com/mfaeezshabbir/rgsplat.git
cd rgsplat
bash scripts/install.sh          # Linux / macOS
powershell -File scripts\install.ps1   # Windows
```

## Quick Start

```bash
# Basic run (CPU)
rgsplat --input my_scene.mp4 --output ./output

# Use a directory of pre-extracted frames instead of a video
rgsplat --input ./frames --output ./output

# Reuse an existing COLMAP workspace under --output (skip ffmpeg + COLMAP)
rgsplat --input ./frames --output ./output --skip-sfm

# Build with the optional GPU projection path
cargo run --release --features gpu -- --input my_scene.mp4 --output ./output --gpu

# View all options
rgsplat --help
```

The binary is named `rgsplat`.

## Pipeline Stages

1. **Frame extraction** - Uses ffmpeg to extract frames from input video
2. **Structure from Motion (SfM)** - Runs COLMAP for camera pose estimation and sparse point cloud generation
3. **Gaussian initialization** - Converts SfM points to 3D Gaussians
4. **Training loop** - Optimizes Gaussian parameters using Adam optimizer with densification/pruning
5. **Export** - Saves trained Gaussians as PLY point cloud

## Requirements

- Rust ≥ 1.85 (edition 2024)
- ffmpeg on `PATH` (for frame extraction from video)
- COLMAP on `PATH` (for SfM camera poses + sparse point cloud)
- Vulkan/Metal/DX12 driver (only for `--features gpu`)
- Python ≥ 3.9 + maturin (only for `--features python`)

## Building

```bash
# CPU-only (default)
cargo build --release

# With GPU support
cargo build --release --features gpu

# With video features
cargo build --release --features video

# Full build
cargo build --release --features full
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Based on the 3D Gaussian Splatting technique introduced by Kerbl et al. (2023).