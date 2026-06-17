//! io/ — input/output: model export and training-data loading.
//!
//! - [`ply`]     — read/write the standard 3DGS `.ply` and `.splat` formats.
//! - [`images`]  — decode training target images into linear `f32` RGBA buffers.
//! - [`cameras`] — NeRF-style `transforms.json` camera import/export.

pub mod ply;
pub mod images;
pub mod cameras;

pub use ply::{load_ply, save_ply, save_splat};
pub use images::{TargetImage, load_images_parallel};
