//! Model sessions: creation, warm-up, inspection (MODELS.md §6, §9).
//!
//! Rules that hold for every model in here:
//!
//! - **One `Session` per model, owned by exactly one thread.** Never a
//!   `Mutex<Session>` in the hot path. Harmless at 3 Hz, fatal at 15 Hz across
//!   four models — and mandatory under DirectML, where only one thread may
//!   call `Run()` on a session.
//! - **Budget ORT's threads against your own.** Each session spins its own
//!   intra-op pool defaulting to roughly core count; four sessions on an
//!   8-core laptop is ~32 threads over 8 cores, which measures *worse* than
//!   single-threaded.
//! - **Warm up.** The first inference is much slower than steady state, so
//!   without warm-up the first real frame is an outlier and calibration starts
//!   on garbage timing.

pub mod face;
pub mod gaze;
pub mod identity;
pub mod objects;
pub mod pose;

use std::path::{Path, PathBuf};

use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::ValueType;

use crate::config::{ExecutionProviderPref, ModelSlot, RuntimeConfig};
use crate::error::{DetectError, Result};

/// Per-stage cost of one model run, in microseconds (MODELS.md §11).
///
/// Three numbers rather than one, because "the model is slow" and "the resize
/// is slow" call for completely different fixes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTimings {
    pub preprocess_us: u32,
    pub inference_us: u32,
    pub postprocess_us: u32,
}

impl StageTimings {
    pub fn total_us(&self) -> u32 {
        self.preprocess_us + self.inference_us + self.postprocess_us
    }
}

/// One tensor's declared name, type and shape. `-1` marks a dynamic axis.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    /// Symbolic name per axis, empty string where the axis is fixed.
    ///
    /// Needed because pinning a dynamic axis with
    /// `SessionBuilder::with_dimension_override` addresses it *by name* — so
    /// "this model has a dynamic dimension" is not actionable on its own, and
    /// the name is not visible anywhere else.
    pub symbols: Vec<String>,
}

impl TensorSpec {
    /// Whether every axis is fixed. The GPU providers want fully static
    /// shapes at session creation, so this is worth knowing before step 5.
    pub fn is_static(&self) -> bool {
        !self.shape.is_empty() && self.shape.iter().all(|d| *d > 0)
    }
}

impl std::fmt::Display for TensorSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A dynamic axis prints its symbolic name rather than a bare `?`,
        // because the name is what `with_dimension_override` needs and there
        // is nowhere else to read it from.
        let dims: Vec<String> = self
            .shape
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if *d >= 0 {
                    return d.to_string();
                }
                match self.symbols.get(i) {
                    Some(sym) if !sym.is_empty() => format!("?{sym}"),
                    _ => "?".to_string(),
                }
            })
            .collect();
        write!(f, "{:<24} {:<8} [{}]", self.name, self.dtype, dims.join(", "))
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

impl ModelInfo {
    pub fn has_dynamic_axes(&self) -> bool {
        self.inputs.iter().chain(&self.outputs).any(|t| !t.is_static())
    }
}

/// Read a model's declared interface without committing to how it will be run.
///
/// Worth doing before writing a single line of pre- or post-processing: the
/// published shape of a model and the shape it actually exports with are not
/// reliably the same thing, and guessing wrong produces plausible-looking
/// garbage rather than an error.
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    let path = path.as_ref();
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let session = (|| -> std::result::Result<Session, ort::Error> {
        let mut builder = Session::builder()?;
        builder.commit_from_file(path)
    })()
    .map_err(|e| model_load(path, e))?;

    Ok(ModelInfo {
        path: path.to_path_buf(),
        size_bytes,
        inputs: session.inputs().iter().map(outlet_spec).collect(),
        outputs: session.outputs().iter().map(outlet_spec).collect(),
    })
}

fn outlet_spec(outlet: &ort::value::Outlet) -> TensorSpec {
    let (dtype, shape, symbols) = match outlet.dtype() {
        ValueType::Tensor { ty, shape, dimension_symbols } => (
            format!("{ty:?}"),
            shape.to_vec(),
            dimension_symbols.iter().map(|s| s.to_string()).collect(),
        ),
        other => (format!("{other:?}"), Vec::new(), Vec::new()),
    };
    TensorSpec { name: outlet.name().to_string(), dtype, shape, symbols }
}

/// Which execution provider a session actually ended up on.
///
/// Reported rather than assumed, because ORT falls back to CPU **silently**
/// when an EP fails to register — and that is the single most misread failure
/// in this stack. It looks exactly like "the GPU didn't help" when the GPU
/// never ran at all. Never infer this from a latency change; read it from the
/// startup log, the HUD or the session report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveEp {
    #[default]
    Cpu,
    DirectMl,
    Cuda,
    CoreMl,
}

impl ActiveEp {
    pub fn as_str(self) -> &'static str {
        match self {
            ActiveEp::Cpu => "CPU",
            ActiveEp::DirectMl => "DirectML",
            ActiveEp::Cuda => "CUDA",
            ActiveEp::CoreMl => "CoreML",
        }
    }
}

/// The GPU execution providers this binary was built with, in the order they
/// are tried. Empty on a CPU build.
///
/// Chosen by Cargo feature (`gpu-directml`, `gpu-cuda`, `gpu-coreml`) and
/// gated on the OS each provider exists for. The gate is not tidiness: the
/// prebuilt ONNX Runtime for Linux has no DirectML in it, and registering
/// DirectML there anyway was a SIGSEGV, not an error (9fc8956). A feature
/// switched on for the wrong target therefore does nothing at all.
///
/// CUDA goes ahead of DirectML when a Windows build has both: on an NVIDIA
/// card it is the faster of the two, and DirectML still catches every other
/// DX12 device.
pub fn gpu_eps() -> Vec<ActiveEp> {
    let mut eps = Vec::new();
    if cfg!(all(feature = "gpu-cuda", any(target_os = "linux", windows))) {
        eps.push(ActiveEp::Cuda);
    }
    if cfg!(all(feature = "gpu-directml", windows)) {
        eps.push(ActiveEp::DirectMl);
    }
    if cfg!(all(feature = "gpu-coreml", target_os = "macos")) {
        eps.push(ActiveEp::CoreMl);
    }
    eps
}

/// Free-dimension overrides needed to give a model a fully static shape.
///
/// The GPU providers want every shape known at session creation. Four of the five
/// models are already static; ArcFace ships with a dynamic batch axis, and
/// rather than re-exporting the weights the axis is pinned here by name at
/// load time. Same result, no modified model file to keep in sync.
type DimOverrides<'a> = &'a [(&'a str, i64)];

/// Build a session with this crate's threading and execution-provider policy.
///
/// `large` selects the bigger intra-op budget — true for YOLO's graph, false
/// for the small high-rate models where sync overhead exceeds the win.
pub fn build_session(path: impl AsRef<Path>, rt: &RuntimeConfig, large: bool) -> Result<Session> {
    build_session_for(path, rt, large, ModelSlot::Face, &[]).map(|(session, _)| session)
}

/// As [`build_session`], but for a named slot — which decides the requested
/// execution provider — and returning which provider was actually obtained.
pub fn build_session_for(
    path: impl AsRef<Path>,
    rt: &RuntimeConfig,
    large: bool,
    slot: ModelSlot,
    dims: DimOverrides<'_>,
) -> Result<(Session, ActiveEp)> {
    let path = path.as_ref();
    let gpu = match rt.providers.for_slot(slot) {
        ExecutionProviderPref::GpuThenCpu => gpu_eps(),
        ExecutionProviderPref::CpuOnly => Vec::new(),
    };

    for ep in gpu {
        match try_build(path, rt, large, ep, dims) {
            Ok(session) => {
                tracing::info!(
                    model = %path.display(),
                    slot = slot.as_str(),
                    ep = ep.as_str(),
                    "session created"
                );
                return Ok((session, ep));
            }
            // Deliberately not fatal, and deliberately loud. A machine without
            // a usable GPU is a supported machine; one that silently ran on
            // the CPU while the log claimed otherwise is not.
            Err(e) => tracing::warn!(
                model = %path.display(),
                slot = slot.as_str(),
                ep = ep.as_str(),
                error = %e,
                "GPU execution provider unavailable for this session; falling back"
            ),
        }
    }

    let session =
        try_build(path, rt, large, ActiveEp::Cpu, dims).map_err(|e| model_load(path, e))?;
    tracing::info!(
        model = %path.display(),
        slot = slot.as_str(),
        ep = "CPU",
        "session created"
    );
    Ok((session, ActiveEp::Cpu))
}

fn try_build(
    path: &Path,
    rt: &RuntimeConfig,
    large: bool,
    provider: ActiveEp,
    dims: DimOverrides<'_>,
) -> std::result::Result<Session, ort::Error> {
    let intra = if large { rt.intra_threads_large } else { rt.intra_threads_small };

    // `SessionBuilder`'s combinators return their own error type carrying the
    // builder back, so this reads better as a sequence than a chain.
    let mut builder = Session::builder()?
        .with_intra_threads(intra)?
        .with_inter_threads(rt.inter_threads)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // ORT's default constant-cost parallelism model produces high
        // latency variance. Decreasing-granularity work claiming trades a
        // little mean for a much tighter tail, and p95 is what a candidate
        // actually experiences as a stutter (MODELS.md §6 rule 2).
        .with_dynamic_block_base(rt.dynamic_block_base as u32)?
        // ORT's thread pools spin-wait between inferences by default,
        // which suits back-to-back batch work and actively harms this
        // pipeline: every worker here is cadence-driven with long idle
        // gaps, so spinning pools burn cores doing nothing and starve the
        // workers that are trying to run. Measured cost of leaving it on:
        // the face worker's p50 tripled once a second session existed.
        .with_intra_op_spinning(rt.allow_spinning)?
        .with_inter_op_spinning(rt.allow_spinning)?;

    // Pin any dynamic axis before the EP sees the graph. The GPU providers
    // partition at session creation, so a dimension left free here is a
    // subgraph that silently stays on the CPU.
    for (name, size) in dims {
        builder = builder.with_dimension_override(*name, *size)?;
    }

    // `error_on_failure` on every provider below is what turns ORT's silent
    // CPU fallback into something `build_session_for` can see and report.
    // Without it the session would build "successfully" on the CPU and every
    // layer above would believe the GPU was running.
    match provider {
        ActiveEp::Cpu => {}
        ActiveEp::DirectMl => {
            builder = builder
                // Both are mandatory for DirectML, not tuning knobs: the EP
                // does not support parallel execution, and its memory-pattern
                // planner conflicts with ORT's. Leaving either at its default
                // produces either an outright failure or quietly wrong
                // behaviour.
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])?;
        }
        ActiveEp::Cuda => {
            builder = builder
                .with_execution_providers([ep::CUDA::default().build().error_on_failure()])?;
        }
        ActiveEp::CoreMl => {
            builder = builder
                .with_execution_providers([ep::CoreML::default().build().error_on_failure()])?;
        }
    }

    // `enable_cpu_mem_arena` is left ON deliberately: disabling it saves
    // memory but ORT's own docs say it increases latency.
    builder.commit_from_file(path)
}

/// What a synthetic benchmark of one model measured.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub size_bytes: u64,
    pub load_ms: f64,
    pub latency: crate::report::LatencySummary,
    pub input_shapes: Vec<Vec<i64>>,
    /// Which EP this measurement was actually taken on. Without it a bench
    /// table comparing CPU and GPU could silently be two CPU columns.
    pub ep: ActiveEp,
}

/// Map a model file to its pipeline slot by conventional filename.
///
/// The benchmark is given a path, not a slot, and the slot decides both the
/// execution provider and whether a dimension override is needed.
fn slot_for_path(path: &Path) -> ModelSlot {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
    if name.contains("yunet") || name.contains("face_detection") {
        ModelSlot::Face
    } else if name.contains("headpose") || name.contains("mobilenetv3") {
        ModelSlot::Pose
    } else if name.contains("gaze") || name.contains("mobileone") {
        ModelSlot::Gaze
    } else if name.contains("yolo") {
        ModelSlot::Objects
    } else {
        ModelSlot::Identity
    }
}

/// Load a model and time it on zero tensors.
///
/// This measures the graph, not the task: no real preprocessing, no decode. It
/// answers "does this file load, and roughly what does a forward pass cost on
/// this machine" — which is what you need when choosing between slots, and is
/// the only per-model number available before the surrounding code exists.
/// Treat it as a floor; the numbers in `live` include preprocessing and are
/// the ones that matter.
pub fn bench_model(path: impl AsRef<Path>, rt: &RuntimeConfig, iters: u32) -> Result<BenchResult> {
    let path = path.as_ref();
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let large = size_bytes > 8 * 1024 * 1024;

    // Infer the slot from the filename so the benchmark uses the same
    // execution provider the pipeline would give this model. Benchmarking
    // every model under one slot's EP setting would answer a question nobody
    // asked.
    let slot = slot_for_path(path);
    let dims = if slot == ModelSlot::Identity { crate::models::identity::ARCFACE_DIMS } else { &[] };

    let t = std::time::Instant::now();
    let (mut session, ep) = build_session_for(path, rt, large, slot, dims)?;
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Allocate a zero buffer per input, substituting 1 for any dynamic axis.
    let mut names = Vec::new();
    let mut shapes: Vec<Vec<i64>> = Vec::new();
    let mut buffers: Vec<Vec<f32>> = Vec::new();
    for outlet in session.inputs() {
        let spec = outlet_spec(outlet);
        let shape: Vec<i64> = spec.shape.iter().map(|d| if *d > 0 { *d } else { 1 }).collect();
        let count: usize = shape.iter().map(|d| *d as usize).product();
        names.push(spec.name);
        shapes.push(shape);
        buffers.push(vec![0.0f32; count]);
    }

    let mut latencies = crate::report::Latencies::with_capacity(iters as usize);
    let total = iters + rt.warmup_iters;
    for i in 0..total {
        let mut inputs: Vec<(std::borrow::Cow<'_, str>, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(names.len());
        for ((name, shape), buf) in names.iter().zip(&shapes).zip(&buffers) {
            let tensor = ort::value::TensorRef::from_array_view((shape.clone(), buf.as_slice()))
                .map_err(|e| DetectError::Inference { model: "bench", source: Box::new(e) })?;
            inputs.push((std::borrow::Cow::from(name.as_str()), tensor.into()));
        }

        let t = std::time::Instant::now();
        session
            .run(inputs)
            .map_err(|e| DetectError::Inference { model: "bench", source: Box::new(e) })?;
        // Discard warm-up: the first inferences are much slower than steady
        // state, and averaging them in hides the number you actually want.
        if i >= rt.warmup_iters {
            latencies.record(t.elapsed());
        }
    }

    Ok(BenchResult {
        name: path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        size_bytes,
        load_ms,
        latency: latencies.summary().unwrap_or_default(),
        input_shapes: shapes,
        ep,
    })
}

fn model_load(path: &Path, e: ort::Error) -> DetectError {
    DetectError::ModelLoad { path: path.to_path_buf(), source: Box::new(e) }
}

/// Zero-copy `[1, 3, side, side]` view over a preallocated NCHW buffer.
///
/// Shared by every square-input model here — reallocating a tensor per frame
/// is pure hot-loop pressure (MODELS.md §6 rule 4), and three copies of this
/// helper would be three places to get the shape wrong.
pub(crate) fn nchw_input<'a>(
    tensor: &'a [f32],
    side: u32,
    stage: &'static str,
) -> Result<ort::value::TensorRef<'a, f32>> {
    ort::value::TensorRef::from_array_view((vec![1i64, 3, side as i64, side as i64], tensor))
        .map_err(|e| inference_error(stage, e))
}

/// Greedy non-maximum suppression, highest score first, **within each class**.
///
/// One implementation for every detector here. `key` yields
/// `(class, bbox, score)`; passing a constant class makes it class-agnostic,
/// which is what a single-class face detector wants. Suppressing across
/// classes would let a confident phone erase a book that overlaps it.
pub(crate) fn nms<T>(
    mut items: Vec<T>,
    iou_threshold: f32,
    top_k: usize,
    key: impl Fn(&T) -> (u32, crate::types::BBox, f32),
) -> Vec<T> {
    items.sort_by(|a, b| {
        key(b).2.partial_cmp(&key(a).2).unwrap_or(std::cmp::Ordering::Equal)
    });
    items.truncate(top_k);

    let mut kept: Vec<T> = Vec::new();
    for candidate in items {
        let (cls, bbox, _) = key(&candidate);
        let suppressed = kept.iter().any(|k| {
            let (kept_cls, kept_bbox, _) = key(k);
            kept_cls == cls && kept_bbox.iou(&bbox) > iou_threshold
        });
        if !suppressed {
            kept.push(candidate);
        }
    }
    kept
}

/// Write a packed RGB8 image into the top-left of a planar `[3, side, side]`
/// tensor, and pad only what the image leaves uncovered.
///
/// Shared by all five models. The tensor is split into its three planes and
/// each source row is zipped against the three destination rows, so every
/// plane is written as one sequential stream with no bounds checks in the
/// inner loop — which is what lets the u8 -> f32 convert vectorise. Filling
/// the whole tensor first and then overwriting most of it cost a full extra
/// pass (4.9 MB per YuNet frame); only the right and bottom bands need it.
///
/// `norm(c, v)` maps source channel `c` (0 = R) to the tensor value. `BGR`
/// puts source channel 2 in plane 0.
#[inline]
pub(crate) fn write_planar<const BGR: bool>(
    tensor: &mut [f32],
    side: usize,
    rgb: &[u8],
    width: usize,
    height: usize,
    pad: f32,
    norm: impl Fn(usize, f32) -> f32,
) {
    let plane = side * side;
    let (p0, rest) = tensor[..3 * plane].split_at_mut(plane);
    let (p1, p2) = rest.split_at_mut(plane);
    let (c0, c2) = if BGR { (2, 0) } else { (0, 2) };

    let mut rows_written = 0;
    if width > 0 {
        let src_rows = rgb.chunks_exact(width * 3).take(height.min(side));
        let dst_rows = p0
            .chunks_exact_mut(side)
            .zip(p1.chunks_exact_mut(side))
            .zip(p2.chunks_exact_mut(side));
        for (src, ((d0, d1), d2)) in src_rows.zip(dst_rows) {
            let (d0, pad0) = d0.split_at_mut(width);
            let (d1, pad1) = d1.split_at_mut(width);
            let (d2, pad2) = d2.split_at_mut(width);
            for (((px, a), b), c) in src.as_chunks::<3>().0.iter().zip(d0).zip(d1).zip(d2) {
                *a = norm(c0, px[c0] as f32);
                *b = norm(1, px[1] as f32);
                *c = norm(c2, px[c2] as f32);
            }
            pad0.fill(pad);
            pad1.fill(pad);
            pad2.fill(pad);
            rows_written += 1;
        }
    }

    let covered = rows_written * side;
    p0[covered..].fill(pad);
    p1[covered..].fill(pad);
    p2[covered..].fill(pad);
}

pub(crate) fn inference_error(model: &'static str, e: ort::Error) -> DetectError {
    DetectError::Inference { model, source: Box::new(e) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_axes_are_reported_as_such() {
        let dynamic = TensorSpec {
            name: "input".into(),
            dtype: "Float32".into(),
            shape: vec![1, 3, -1, -1],
            symbols: vec![String::new(), String::new(), String::new(), String::new()],
        };
        assert!(!dynamic.is_static());
        assert!(dynamic.to_string().contains("[1, 3, ?, ?]"));

        let fixed = TensorSpec {
            name: "input".into(),
            dtype: "Float32".into(),
            shape: vec![1, 3, 320, 320],
            symbols: vec![String::new(); 4],
        };
        assert!(fixed.is_static());
    }

    #[test]
    fn a_named_dynamic_axis_prints_its_name() {
        // The name is what `with_dimension_override` addresses, and there is
        // nowhere else to read it from. ArcFace's batch axis really is called
        // `None` — a PyTorch export artifact — and overriding a name the graph
        // does not use is a silent no-op, so this has to be visible.
        let spec = TensorSpec {
            name: "input.1".into(),
            dtype: "Float32".into(),
            shape: vec![-1, 3, 112, 112],
            symbols: vec!["None".into(), String::new(), String::new(), String::new()],
        };
        assert!(!spec.is_static());
        assert!(spec.to_string().contains("?None"), "got {spec}");
    }

    #[test]
    fn gpu_providers_are_only_offered_where_they_exist() {
        // Registering DirectML on Linux was a SIGSEGV, not an error, so the
        // feature gate is the only thing standing between a mis-built binary
        // and a crash at model load.
        for ep in gpu_eps() {
            match ep {
                ActiveEp::DirectMl => assert!(cfg!(windows)),
                ActiveEp::CoreMl => assert!(cfg!(target_os = "macos")),
                ActiveEp::Cuda => assert!(cfg!(any(target_os = "linux", windows))),
                ActiveEp::Cpu => panic!("CPU is the fallback, not a GPU provider"),
            }
        }
    }

    #[test]
    fn planar_write_swaps_channels_and_pads_only_the_uncovered_bands() {
        // 2x1 image into a 3x3 tensor: pixel (0,0) = (1,2,3), (1,0) = (4,5,6).
        let rgb = [1u8, 2, 3, 4, 5, 6];
        let mut t = vec![-1.0f32; 27];
        write_planar::<true>(&mut t, 3, &rgb, 2, 1, 9.0, |_, v| v);
        // Plane 0 is B, plane 2 is R; everything not covered is the pad.
        assert_eq!(&t[0..9], &[3.0, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
        assert_eq!(&t[9..18], &[2.0, 5.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
        assert_eq!(&t[18..27], &[1.0, 4.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);

        write_planar::<false>(&mut t, 3, &rgb, 2, 1, 0.0, |c, v| v * 10.0 + c as f32);
        assert_eq!(&t[0..2], &[10.0, 40.0]);
        assert_eq!(&t[18..20], &[32.0, 62.0]);
    }

    #[test]
    fn a_rankless_tensor_is_not_static() {
        // An empty shape means ORT told us nothing useful; treating that as
        // "static" would let it silently reach a GPU provider at step 5.
        let spec = TensorSpec {
            name: "x".into(),
            dtype: "Float32".into(),
            shape: vec![],
            symbols: vec![],
        };
        assert!(!spec.is_static());
    }
}
