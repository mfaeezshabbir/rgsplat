//! io/ply.rs — read/write 3D Gaussian Splatting models.
//!
//! Two formats are supported:
//!
//! * **`.ply`** — the de-facto 3DGS interchange format used by the INRIA
//!   reference code and most viewers. Per vertex: `x y z`, `nx ny nz`,
//!   `f_dc_0..2` (SH degree-0 RGB), `f_rest_0..44` (higher SH, channel-major),
//!   `opacity` (logit), `scale_0..2` (log), `rot_0..3` (`w,x,y,z` quaternion).
//!   We read/write `binary_little_endian` (and parse ASCII on read).
//! * **`.splat`** — antimatter15's compact 32-byte-per-splat runtime format:
//!   `position[3]:f32`, `scale[3]:f32`, `rgba[4]:u8`, `quat[4]:u8`.
//!
//! Our in-memory [`Gaussian3D`] already stores the *raw* (log/logit) parameters,
//! so `.ply` round-trips losslessly.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::math::{Gaussian3D, sigmoid};

/// SH degree-0 basis constant `1 / (2√π)`.
const SH_C0: f32 = 0.282_094_8;

/// Number of SH coefficients per channel stored in [`Gaussian3D`] (degree 3).
const SH_PER_CHANNEL: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// PLY writer
// ─────────────────────────────────────────────────────────────────────────────

/// Write Gaussians to a binary-little-endian `.ply` in the standard 3DGS layout.
pub fn save_ply(gaussians: &[Gaussian3D], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = File::create(path).with_context(|| format!("creating {path:?}"))?;
    let mut w = BufWriter::new(file);

    // ── Header ────────────────────────────────────────────────────────────────
    writeln!(w, "ply")?;
    writeln!(w, "format binary_little_endian 1.0")?;
    writeln!(w, "element vertex {}", gaussians.len())?;
    for p in ["x", "y", "z", "nx", "ny", "nz"] {
        writeln!(w, "property float {p}")?;
    }
    for i in 0..3 {
        writeln!(w, "property float f_dc_{i}")?;
    }
    // 45 = (16 - 1) coeffs × 3 channels.
    for i in 0..(SH_PER_CHANNEL - 1) * 3 {
        writeln!(w, "property float f_rest_{i}")?;
    }
    writeln!(w, "property float opacity")?;
    for i in 0..3 {
        writeln!(w, "property float scale_{i}")?;
    }
    for i in 0..4 {
        writeln!(w, "property float rot_{i}")?;
    }
    writeln!(w, "end_header")?;

    // ── Body ───────────────────────────────────────────────────────────────────
    let mut buf = Vec::<u8>::with_capacity(62 * 4);
    for g in gaussians {
        buf.clear();
        let mut put = |v: f32| buf.extend_from_slice(&v.to_le_bytes());

        put(g.position[0]);
        put(g.position[1]);
        put(g.position[2]);
        put(0.0); // nx
        put(0.0); // ny
        put(0.0); // nz

        // f_dc: degree-0 coefficient for R, G, B.
        put(g.sh_coeffs[0]);
        put(g.sh_coeffs[SH_PER_CHANNEL]);
        put(g.sh_coeffs[2 * SH_PER_CHANNEL]);

        // f_rest: channel-major (all R higher, then G, then B), matching INRIA.
        for ch in 0..3 {
            for k in 1..SH_PER_CHANNEL {
                put(g.sh_coeffs[ch * SH_PER_CHANNEL + k]);
            }
        }

        put(g.opacity_logit);
        put(g.log_scale[0]);
        put(g.log_scale[1]);
        put(g.log_scale[2]);
        put(g.rotation[0]);
        put(g.rotation[1]);
        put(g.rotation[2]);
        put(g.rotation[3]);

        w.write_all(&buf)?;
    }
    w.flush()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// PLY reader
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Format {
    BinaryLe,
    BinaryBe,
    Ascii,
}

/// Load Gaussians from a 3DGS `.ply`. Tolerant of any property ordering and of
/// SH degree (`f_rest` may have 0/9/24/45 entries); unknown properties are
/// skipped. Only `float`/`double` scalar properties are supported.
pub fn load_ply(path: &Path) -> Result<Vec<Gaussian3D>> {
    let file = File::open(path).with_context(|| format!("opening {path:?}"))?;
    let mut reader = BufReader::new(file);

    // ── Parse header ────────────────────────────────────────────────────────────
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        bail!("not a PLY file (missing magic): {path:?}");
    }

    let mut format = Format::Ascii;
    let mut count = 0usize;
    let mut props: Vec<(String, usize)> = Vec::new(); // (name, byte width)

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            bail!("unexpected EOF in PLY header");
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        match toks.as_slice() {
            ["format", fmt, _] => {
                format = match *fmt {
                    "binary_little_endian" => Format::BinaryLe,
                    "binary_big_endian" => Format::BinaryBe,
                    "ascii" => Format::Ascii,
                    other => bail!("unsupported PLY format: {other}"),
                };
            }
            ["element", "vertex", n] => count = n.parse().context("vertex count")?,
            ["element", ..] => { /* other elements: ignored */ }
            ["property", ty, name] => {
                let width = match *ty {
                    "float" | "float32" => 4,
                    "double" | "float64" => 8,
                    "uchar" | "uint8" | "char" | "int8" => 1,
                    "short" | "ushort" | "int16" | "uint16" => 2,
                    "int" | "uint" | "int32" | "uint32" => 4,
                    other => bail!("unsupported PLY property type: {other}"),
                };
                props.push((name.to_string(), width));
            }
            ["end_header"] => break,
            _ => { /* comment / obj_info / etc. */ }
        }
    }

    // ── Read body ────────────────────────────────────────────────────────────────
    let mut out = Vec::with_capacity(count);

    if format == Format::Ascii {
        for _ in 0..count {
            line.clear();
            reader.read_line(&mut line)?;
            let vals: Vec<f32> = line
                .split_whitespace()
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect();
            out.push(gaussian_from_props(&props, |i| vals.get(i).copied().unwrap_or(0.0)));
        }
    } else {
        let le = format == Format::BinaryLe;
        let mut rec = vec![0u8; props.iter().map(|(_, w)| w).sum()];
        for _ in 0..count {
            reader.read_exact(&mut rec)?;
            let mut off = 0usize;
            // Decode each property to f32 according to its declared width.
            let mut decoded = Vec::with_capacity(props.len());
            for (_, width) in &props {
                let v = decode_scalar(&rec[off..off + width], *width, le);
                decoded.push(v);
                off += width;
            }
            out.push(gaussian_from_props(&props, |i| decoded[i]));
        }
    }

    Ok(out)
}

fn decode_scalar(bytes: &[u8], width: usize, le: bool) -> f32 {
    match width {
        4 => {
            let a: [u8; 4] = bytes.try_into().unwrap();
            if le { f32::from_le_bytes(a) } else { f32::from_be_bytes(a) }
        }
        8 => {
            let a: [u8; 8] = bytes.try_into().unwrap();
            (if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }) as f32
        }
        1 => bytes[0] as f32,
        2 => {
            let a: [u8; 2] = bytes.try_into().unwrap();
            (if le { u16::from_le_bytes(a) } else { u16::from_be_bytes(a) }) as f32
        }
        _ => 0.0,
    }
}

/// Assemble a [`Gaussian3D`] from named PLY properties via an indexed accessor.
fn gaussian_from_props(props: &[(String, usize)], get: impl Fn(usize) -> f32) -> Gaussian3D {
    let mut g = Gaussian3D::new([0.0, 0.0, 0.0]);
    // Default identity rotation in case rot_* are absent.
    g.rotation = [1.0, 0.0, 0.0, 0.0];

    for (i, (name, _)) in props.iter().enumerate() {
        let v = get(i);
        match name.as_str() {
            "x" => g.position[0] = v,
            "y" => g.position[1] = v,
            "z" => g.position[2] = v,
            "f_dc_0" => g.sh_coeffs[0] = v,
            "f_dc_1" => g.sh_coeffs[SH_PER_CHANNEL] = v,
            "f_dc_2" => g.sh_coeffs[2 * SH_PER_CHANNEL] = v,
            "opacity" => g.opacity_logit = v,
            "scale_0" => g.log_scale[0] = v,
            "scale_1" => g.log_scale[1] = v,
            "scale_2" => g.log_scale[2] = v,
            "rot_0" => g.rotation[0] = v,
            "rot_1" => g.rotation[1] = v,
            "rot_2" => g.rotation[2] = v,
            "rot_3" => g.rotation[3] = v,
            other => {
                if let Some(idx) = other.strip_prefix("f_rest_").and_then(|s| s.parse::<usize>().ok())
                {
                    // f_rest is channel-major with `per_ch` coeffs per channel.
                    let per_ch = SH_PER_CHANNEL - 1;
                    let ch = idx / per_ch;
                    let k = idx % per_ch;
                    if ch < 3 {
                        g.sh_coeffs[ch * SH_PER_CHANNEL + 1 + k] = v;
                    }
                }
            }
        }
    }
    g
}

// ─────────────────────────────────────────────────────────────────────────────
// .splat writer (compact runtime format)
// ─────────────────────────────────────────────────────────────────────────────

/// Write the compact 32-byte-per-splat `.splat` format (antimatter15).
///
/// Colors are baked from the SH degree-0 term and opacity is squashed through a
/// sigmoid — this is a *display* format and is intentionally lossy.
pub fn save_splat(gaussians: &[Gaussian3D], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = File::create(path).with_context(|| format!("creating {path:?}"))?;
    let mut w = BufWriter::new(file);

    let mut rec = [0u8; 32];
    for g in gaussians {
        let scale = g.scale();
        let mut off = 0;
        for v in g.position {
            rec[off..off + 4].copy_from_slice(&v.to_le_bytes());
            off += 4;
        }
        for v in scale {
            rec[off..off + 4].copy_from_slice(&v.to_le_bytes());
            off += 4;
        }
        // Color: degree-0 SH → RGB, opacity via sigmoid.
        let to_u8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
        rec[off] = to_u8(0.5 + g.sh_coeffs[0] * SH_C0);
        rec[off + 1] = to_u8(0.5 + g.sh_coeffs[SH_PER_CHANNEL] * SH_C0);
        rec[off + 2] = to_u8(0.5 + g.sh_coeffs[2 * SH_PER_CHANNEL] * SH_C0);
        rec[off + 3] = to_u8(sigmoid(g.opacity_logit));
        off += 4;
        // Rotation: normalized quaternion packed to bytes (q * 128 + 128).
        let q = crate::math::quaternion::normalize(g.rotation);
        for v in q {
            rec[off] = ((v * 128.0 + 128.0).clamp(0.0, 255.0)).round() as u8;
            off += 1;
        }
        w.write_all(&rec)?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ply_roundtrip_preserves_parameters() {
        let mut g = Gaussian3D::new([1.0, -2.0, 3.5]);
        g.opacity_logit = 0.42;
        g.log_scale = [-1.0, -2.0, -0.5];
        g.rotation = crate::math::quaternion::normalize([0.3, 0.7, -0.2, 0.5]);
        for i in 0..48 {
            g.sh_coeffs[i] = (i as f32) * 0.01 - 0.2;
        }
        let original = vec![g.clone(), Gaussian3D::new([0.0, 0.0, 0.0])];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.ply");
        save_ply(&original, &path).unwrap();
        let loaded = load_ply(&path).unwrap();

        assert_eq!(loaded.len(), original.len());
        let (a, b) = (&original[0], &loaded[0]);
        for i in 0..3 {
            assert!((a.position[i] - b.position[i]).abs() < 1e-5);
            assert!((a.log_scale[i] - b.log_scale[i]).abs() < 1e-5);
        }
        assert!((a.opacity_logit - b.opacity_logit).abs() < 1e-5);
        for i in 0..48 {
            assert!((a.sh_coeffs[i] - b.sh_coeffs[i]).abs() < 1e-5, "sh[{i}] mismatch");
        }
    }

    #[test]
    fn splat_record_is_32_bytes() {
        let g = vec![Gaussian3D::new([0.0, 0.0, 0.0]); 10];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.splat");
        save_splat(&g, &path).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 32 * 10);
    }
}
