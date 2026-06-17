//! python.rs — optional Python bindings (built with maturin, behind `python`).
//!
//! Exposes a thin wrapper around the pipeline so it can be driven from Python:
//!
//! ```python
//! import gaussian_splat_pipeline as gsp
//! gsp.run_pipeline("scene.mp4", "./out", iterations=30000, gpu=False)
//! ```

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::pipeline::{Pipeline, PipelineConfig};

fn to_py<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Run the full pipeline (frames → SfM → train → export) and return the output
/// `.ply` path. Blocks until training completes.
#[pyfunction]
#[pyo3(signature = (input, output, iterations = 30_000, gpu = false, fps = 2.0))]
fn run_pipeline(
    py: Python<'_>,
    input: String,
    output: String,
    iterations: usize,
    gpu: bool,
    fps: f32,
) -> PyResult<String> {
    // Release the GIL: this is a long, CPU-bound native computation.
    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new().map_err(to_py)?;
        rt.block_on(async {
            let mut cfg = PipelineConfig::from_video(&input, &output);
            cfg.train.num_iterations = iterations;
            cfg.use_gpu = gpu;
            cfg.extract_fps = fps;
            let pipeline = Pipeline::new(cfg).await.map_err(to_py)?;
            let out = pipeline.run().await.map_err(to_py)?;
            Ok(out.to_string_lossy().into_owned())
        })
    })
}

/// Convert a trained `.ply` to the compact `.splat` runtime format.
#[pyfunction]
fn ply_to_splat(input: String, output: String) -> PyResult<usize> {
    let gaussians = crate::io::load_ply(std::path::Path::new(&input)).map_err(to_py)?;
    crate::io::save_splat(&gaussians, std::path::Path::new(&output)).map_err(to_py)?;
    Ok(gaussians.len())
}

#[pymodule]
fn gaussian_splat_pipeline(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(ply_to_splat, m)?)?;
    Ok(())
}
