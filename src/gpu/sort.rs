//! gpu/sort.rs — depth ordering for projected splats.
//!
//! The GPU projection pass emits one `SplatInfo2D` per Gaussian plus a `u32`
//! depth key. Correct alpha compositing needs them ordered nearest-first; we do
//! that ordering on the CPU here (a full GPU radix sort is a worthwhile future
//! optimization but unnecessary for correctness).

use super::compute::SplatInfo2D;

/// Return indices of visible splats ordered nearest-first (ascending depth).
pub fn depth_order(splats: &[SplatInfo2D]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..splats.len())
        .filter(|&i| splats[i].alpha >= 1.0 / 255.0 && splats[i].depth > 0.0)
        .collect();
    order.sort_unstable_by(|&a, &b| splats[a].depth.partial_cmp(&splats[b].depth).unwrap());
    order
}
