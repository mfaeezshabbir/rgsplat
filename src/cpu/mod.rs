//! cpu/ — CPU rendering path.
//!
//! - [`rasterizer`] — tile-based forward rasterizer (Rayon parallel).
//! - [`backward`]   — analytic differentiable rasterizer producing per-Gaussian
//!   gradients, plus the L1 + D-SSIM photometric loss.

pub mod rasterizer;
pub mod backward;

pub use rasterizer::{Framebuffer, rasterize};
