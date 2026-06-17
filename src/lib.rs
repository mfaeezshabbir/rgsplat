//! # rgsplat
//!
//! A Rust implementation of 3D Gaussian Splatting for real-time radiance field
//! rendering (Kerbl et al., SIGGRAPH 2023). The crate is organized as:
//!
//! - [`math`]   — core primitives ([`Gaussian3D`], [`Camera`], SH, projection)
//! - [`cpu`]    — CPU tile rasterizer (forward) and analytic backward pass
//! - [`gpu`]    — optional wgpu compute path (behind the `gpu` feature)
//! - [`io`]     — PLY / `.splat` export, COLMAP & image loading
//! - [`pipeline`] — async orchestration: frames → SfM → init → train → export
//!
//! The whole pipeline can be driven through [`pipeline::Pipeline`].

pub mod math;
pub mod cpu;
pub mod gpu;
pub mod io;
pub mod pipeline;

#[cfg(feature = "python")]
pub mod python;

// ── Convenience re-exports ────────────────────────────────────────────────────
pub use math::{Camera, Gaussian3D};
pub use pipeline::{Pipeline, PipelineConfig};
