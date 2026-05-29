// DeepScreen Viewer frontend.
//
// This file renders what `Signals` contains and nothing more. There is no
// threshold, no hold timer, no hysteresis and no "is this a violation"
// reasoning anywhere in it — all of that belongs to the fusion layer, behind
// one `Config`. CONTEXT.md §11 is what happens when the same constant lives in
// three places with three different values.
//
// It also does no per-frame work: the MJPEG stream is decoded and painted by
// the browser. This file only polls a few hundred bytes of JSON.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const SVG_NS = "http://www.w3.org/2000/svg";
const POLL_MS = 33; // ~30 Hz, which is faster than detection produces

const el = {
  stream: document.getElementById("stream"),
  overlay: document.getElementById("overlay"),
  hud: document.getElementById("hud-lines"),
  error: document.getElementById("error"),
  degraded: document.getElementById("flag-degraded"),
  enrol: document.getElementById("enrol"),
  enrolState: document.getElementById("enrol-state"),
  logRows: document.getElementById("log-rows"),
  setup: document.getElementById("setup"),
  setupText: document.getElementById("setup-text"),
  dirFrame: document.getElementById("dir-frame"),
  dirHeadH: document.getElementById("dir-head-h"),
  dirHeadV: document.getElementById("dir-head-v"),
  dirGazeH: document.getElementById("dir-gaze-h"),
  dirGazeV: document.getElementById("dir-gaze-v"),
  dirEyeH: document.getElementById("dir-eye-h"),
  dirEyeV: document.getElementById("dir-eye-v"),
};

// Pill id per violation kind, using the `ViolationKind::as_str` names exactly
// as Rust emits them. A kind with no pill simply does not get one; the log
// still records it.
const PILLS = {
  no_face: "flag-noface",
  never_seen: "flag-neverseen",
  multiple_faces: "flag-multiface",
  prohibited_object: "flag-object",
  head_turned_away: "flag-head",
  gaze_off_screen: "flag-gaze",
  identity_mismatch: "flag-identity",
  signal_lost: "flag-lost",
};

let sawDegradedEvent = false;
let viewBox = "";

// The object worker runs at 1 Hz against a 15 Hz face worker, and the library
// now reports its results on the one frame they were computed for — never
// carried forward, because "there was a phone 900 ms ago" is an inference and
// inference belongs to fusion.
//
// For *display* that would mean boxes flashing for a single frame per second.
// So the last real result is held here, in the front end, where it is clearly
// a rendering choice and cannot be mistaken for a measurement. Nothing derived
// from this is sent anywhere or used to decide anything.
let lastObjects = { detections: [], seq: -1 };

// Events are edge-triggered and low-rate — the opposite of the polled
// snapshot. Fusion is their producer.
//
// The log is built here rather than from `snapshot` on purpose: a violation is
// a discrete thing that happened at a time, and a row once written never
// changes except to gain its duration. Polling a growing list thirty times a
// second to redraw unchanged rows is precisely the frame-rate re-render the
// whole IPC design exists to prevent.
listen("detection:event", (event) => {
  const payload = event.payload;
  if (!payload) return;
  const kind = String(payload.event || "");

  if (kind === "degraded") {
    sawDegradedEvent = true;
    return;
  }
  if (kind === "recovered") {
    sawDegradedEvent = false;
    return;
  }
  if (kind === "violation_started") {
    appendViolation(payload);
  } else if (kind === "violation_ended") {
    closeViolation(payload);
  }
});

// Every string below is Rust's own: `kind`, `severity` and `subject` are
// printed verbatim. No threshold, no comparison and no severity decision
// happens in this file.
function rowId(v) {
  return `v-${v.kind}-${v.subject || ""}-${v.t_start_ms}`;
}

function secs(ms) {
  const t = (ms || 0) / 1000;
  const m = Math.floor(t / 60);
  return `${m}:${String((t % 60).toFixed(1)).padStart(4, "0")}`;
}

function appendViolation(v) {
  const row = document.createElement("div");
  row.className = `log-row sev-${v.severity}`;
  row.id = rowId(v);
  row.innerHTML =
    `<b>${secs(v.t_start_ms)}</b> <i>${v.kind}</i>` +
    `${v.subject ? ` <u>${v.subject}</u>` : ""}` +
    `<em class="sev">${v.severity}</em><span class="dur">ongoing</span>`;
  el.logRows.prepend(row);

  // Unbounded growth over a three-hour exam is a leak with a UI. The full
  // record lives in Rust; this is a window onto the recent end of it.
  while (el.logRows.childElementCount > 200) {
    el.logRows.lastElementChild.remove();
  }
}

function closeViolation(v) {
  const row = document.getElementById(rowId(v));
  if (!row) return;
  const dur = ((v.t_end_ms || v.t_start_ms) - v.t_start_ms) / 1000;
  row.querySelector(".dur").textContent = `${dur.toFixed(1)}s`;
  row.classList.add("ended");
}

async function boot() {
  const port = await invoke("stream_port");
  if (port > 0) {
    el.stream.src = `http://127.0.0.1:${port}/stream`;
  }
  setInterval(poll, POLL_MS);
}

async function poll() {
  let snap;
  try {
    snap = await invoke("snapshot");
  } catch (e) {
    return; // a dropped poll is not worth reporting; the next one is 33 ms away
  }

  // A missing prerequisite takes over the whole window. It is not an error
  // strip over a dead video pane — there is nothing behind it to look at, and
  // the only useful thing on screen is the instruction.
  if (snap.setup_blocked) {
    el.setup.hidden = false;
    el.setupText.textContent = snap.error || "";
    el.error.hidden = true;
    return;
  }
  el.setup.hidden = true;

  if (snap.error) {
    el.error.hidden = false;
    el.error.textContent = snap.error;
  } else {
    el.error.hidden = true;
  }

  // Refresh the held object result before anything draws, so the overlay, the
  // pill and the HUD all describe the same thing.
  if (snap.signals?.produced_by?.objects === "produced") {
    lastObjects = { detections: snap.signals.objects || [], seq: snap.seq };
  }

  drawOverlay(snap);
  drawHud(snap);
  drawFlags(snap);
  drawDirections(snap);
}

// Prints strings the library already decided. There is no angle here, no
// threshold and no comparison against one — bucketing lives in Rust, in
// `direction.rs`, with the hysteresis that stops these words flickering.
//
// The only test in this function is `=== "CENTER"`, and it chooses a colour.
// Dimming the resting state is presentation; it decides nothing.
function drawDirections(snap) {
  const d = snap.signals?.debug_directions;

  // The frame of reference arrives as prose ("subject POV") so this cannot
  // drift from what the library actually meant. Mirroring is the viewer's own
  // business, so it is reported here rather than by the library.
  const mirrored = snap.preview_mirrored ? "mirrored" : "not mirrored";
  el.dirFrame.textContent = d ? `${d.frame_of_reference} · ${mirrored}` : "—";

  // `null` is "this signal did not run", which must not look like CENTER.
  const put = (node, value) => {
    node.textContent = value ?? "—";
    node.classList.toggle("none", !value);
    node.classList.toggle("centre", value === "CENTER");
  };

  put(el.dirHeadH, d?.head?.horizontal);
  put(el.dirHeadV, d?.head?.vertical);
  put(el.dirGazeH, d?.gaze?.horizontal);
  put(el.dirGazeV, d?.gaze?.vertical);
  put(el.dirEyeH, d?.eye?.horizontal);
  put(el.dirEyeV, d?.eye?.vertical);
}

function drawOverlay(snap) {
  // The viewBox is the SOURCE resolution, so bbox coordinates go in
  // unmodified and the browser does the scaling.
  if (snap.width > 0 && snap.height > 0) {
    const wanted = `0 0 ${snap.width} ${snap.height}`;
    if (wanted !== viewBox) {
      el.overlay.setAttribute("viewBox", wanted);
      viewBox = wanted;
    }
  }

  el.overlay.replaceChildren();
  const s = snap.signals || {};

  for (const face of s.faces || []) {
    const b = face.bbox;
    el.overlay.appendChild(rect(b, "#22c55e"));
    el.overlay.appendChild(
      label(b.x, b.y - 6, `${(face.score * 100).toFixed(0)}%`, "#22c55e"),
    );

    // Keypoint order is fixed by the model: eyes, nose, mouth corners.
    const k = face.keypoints;
    if (k) {
      for (const [p, colour] of [
        [k.right_eye, "#22c55e"],
        [k.left_eye, "#22c55e"],
        [k.nose, "#facc15"],
        [k.right_mouth, "#f87171"],
        [k.left_mouth, "#f87171"],
      ]) {
        el.overlay.appendChild(dot(p[0], p[1], colour));
      }
    }
  }

  // Head-pose gizmo on the primary face. Projection ported from the model
  // author's own `draw_axis`, including the yaw negation on the first line —
  // deriving this independently is how you get a gizmo that looks right and
  // rotates the wrong way.
  const pose = s.head_pose;
  if (pose && (s.faces || []).length > 0) {
    const b = s.faces[0].bbox;
    drawPoseGizmo(pose, b);
  }

  // Gaze ray from between the eyes. Projection matches the reference
  // implementation: +yaw points left of screen, +pitch points up.
  const gaze = s.gaze;
  if (gaze && (s.faces || []).length > 0) {
    drawGazeRay(gaze, s.faces[0]);
  }

  // Held result, not this frame's — see `lastObjects`.
  for (const obj of lastObjects.detections) {
    el.overlay.appendChild(rect(obj.bbox, "#ef4444"));
    el.overlay.appendChild(
      label(
        obj.bbox.x,
        obj.bbox.y - 6,
        `${obj.label} ${(obj.score * 100).toFixed(0)}%`,
        "#ef4444",
      ),
    );
  }
}

function drawGazeRay(gaze, face) {
  const k = face.keypoints;
  const b = face.bbox;
  // Between the eyes when keypoints exist, box centre otherwise.
  const ox = k ? (k.left_eye[0] + k.right_eye[0]) / 2 : b.x + b.w / 2;
  const oy = k ? (k.left_eye[1] + k.right_eye[1]) / 2 : b.y + b.h / 2;

  const len = b.w * 1.1;
  const dx = -len * Math.sin(gaze.yaw_rad) * Math.cos(gaze.pitch_rad);
  const dy = -len * Math.sin(gaze.pitch_rad);

  el.overlay.appendChild(line(ox, oy, ox + dx, oy + dy, "#f0abfc"));
  el.overlay.appendChild(dot(ox, oy, "#f0abfc"));
}

function drawPoseGizmo(pose, b) {
  const rad = Math.PI / 180;
  const yaw = -pose.yaw_deg * rad; // negated, as the reference does
  const pitch = pose.pitch_deg * rad;
  const roll = pose.roll_deg * rad;

  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cr = Math.cos(roll), sr = Math.sin(roll);

  const tdx = b.x + b.w * 0.5;
  const tdy = b.y + b.h * 0.5;
  const size = Math.min(b.w, b.h) * 0.5;

  const axes = [
    [size * (cy * cr), size * (cp * sr + cr * sp * sy), "#ef4444"], // X right
    [size * (-cy * sr), size * (cp * cr - sp * sy * sr), "#22c55e"], // Y down
    [size * sy, size * (-cy * sp), "#60a5fa"], // Z out of screen
  ];
  for (const [dx, dy, colour] of axes) {
    el.overlay.appendChild(line(tdx, tdy, tdx + dx, tdy + dy, colour));
  }
}

function line(x1, y1, x2, y2, colour) {
  const l = document.createElementNS(SVG_NS, "line");
  l.setAttribute("x1", x1);
  l.setAttribute("y1", y1);
  l.setAttribute("x2", x2);
  l.setAttribute("y2", y2);
  l.setAttribute("stroke", colour);
  l.setAttribute("stroke-width", "4");
  l.setAttribute("stroke-linecap", "round");
  return l;
}

function rect(b, colour) {
  const r = document.createElementNS(SVG_NS, "rect");
  r.setAttribute("x", b.x);
  r.setAttribute("y", b.y);
  r.setAttribute("width", b.w);
  r.setAttribute("height", b.h);
  r.setAttribute("fill", "none");
  r.setAttribute("stroke", colour);
  r.setAttribute("stroke-width", "3");
  return r;
}

function dot(x, y, colour) {
  const c = document.createElementNS(SVG_NS, "circle");
  c.setAttribute("cx", x);
  c.setAttribute("cy", y);
  c.setAttribute("r", "3.5");
  c.setAttribute("fill", colour);
  return c;
}

function label(x, y, text, colour) {
  const t = document.createElementNS(SVG_NS, "text");
  t.setAttribute("x", x);
  t.setAttribute("y", y);
  t.setAttribute("fill", colour);
  t.setAttribute("font-size", "18");
  t.setAttribute("font-family", "ui-monospace, Consolas, monospace");
  t.textContent = text;
  return t;
}

function drawHud(snap) {
  const st = snap.stats || {};
  const ms = (us) => (us / 1000).toFixed(1);
  const faces = (snap.signals?.faces || []).length;
  // Held count, matching the boxes and the pill. Its freshness is the
  // `objects:` entry on the slots line below.
  const objects = lastObjects.detections.length;

  el.hud.textContent = [
    `${snap.source}`,
    `cap ${fmt(st.capture_fps)} fps   det ${fmt(st.detect_fps)} fps   skipped ${st.frames_skipped ?? 0}`,
    `detect  p50 ${ms(st.detect_p50_us ?? 0)} ms   p95 ${ms(st.detect_p95_us ?? 0)} ms`,
    `preview p50 ${ms(snap.preview_p50_us ?? 0)} ms`,
    `faces ${faces}   objects ${objects}   seq ${snap.seq}`,
    poseLine(snap.signals?.head_pose),
    gazeLine(snap.signals?.gaze),
    slotLine(snap.signals?.produced_by),
    // Permanent, not a probe: a viewport wider than the window silently crops
    // the frame and pushes right-anchored UI off-screen. An instrument should
    // report the geometry it is drawing into.
    `view ${window.innerWidth}x${window.innerHeight}  dpr ${window.devicePixelRatio}`,
    // Alongside the other geometry, because that is what it is. A flipped
    // preview and a backwards label look identical until one of them is
    // written down.
    `mirrored ${snap.preview_mirrored ? "yes" : "no"}`,
    snap.running ? "" : "— source ended —",
  ]
    .filter(Boolean)
    .join("\n");
}

function gazeLine(gaze) {
  if (!gaze) return "gaze    —";
  const deg = (r) => (r * 180) / Math.PI;
  const f = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);
  const head = `gaze    yaw ${f(deg(gaze.yaw_rad))}  pitch ${f(deg(gaze.pitch_rad))}`;
  if (gaze.eye_yaw_rad === null || gaze.eye_yaw_rad === undefined) return head;
  // Eye-in-head: gaze minus head pose. This is the eyeball-movement signal —
  // large here with the pose gizmo still means eyes moving independently.
  return (
    head +
    `
eye     yaw ${f(deg(gaze.eye_yaw_rad))}  pitch ${f(deg(gaze.eye_pitch_rad))}`
  );
}

// What each model slot did on THIS frame. The states come from the library
// verbatim; this only lays them out.
//
// Worth having on screen permanently: "gaze absent" and "gaze centred" look
// identical in the angle readout above, and for a proctoring system reading
// the first as the second is a false negative — the failure mode that matters
// most. When gaze is gated, the reason is appended.
function slotLine(coverage) {
  if (!coverage) return "slots   —";
  const slots = ["face", "pose", "gaze", "objects"]
    .map((k) => `${k}:${coverage[k] ?? "?"}`)
    .join(" ");
  const why = coverage.gaze_gate ? `  (${coverage.gaze_gate})` : "";
  return `slots   ${slots}${why}`;
}

function poseLine(pose) {
  if (!pose) return "pose    —";
  const f = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);
  return `pose    yaw ${f(pose.yaw_deg)}  pitch ${f(pose.pitch_deg)}  roll ${f(pose.roll_deg)}`;
}

function fmt(v) {
  return (v ?? 0).toFixed(1);
}

// Pills reflect fusion's decisions, not raw per-frame signals.
//
// They used to be `faces.length === 0` and friends: instantaneous, with no
// memory, flickering on every dropped detection. That was honest about being
// raw and useless as a violation display. What lights now has already survived
// a hold timer and hysteresis in Rust — and this function still decides
// nothing, it only reads a list of names.
function drawFlags(snap) {
  const active = new Set(snap.active_violations || []);
  for (const [kind, id] of Object.entries(PILLS)) {
    const pill = document.getElementById(id);
    if (pill) pill.classList.toggle("on", active.has(kind));
  }
  const degraded = sawDegradedEvent || (snap.degraded || []).length > 0;
  el.degraded.classList.toggle("on", degraded);

  el.enrolState.textContent = snap.enrolled ? "enrolled" : "not enrolled";
}

el.enrol.addEventListener("click", async () => {
  el.enrol.disabled = true;
  el.enrol.textContent = "enrolling…";
  try {
    await invoke("enrol");
  } finally {
    // The worker enrols on its next cycle, up to 5 s away at 0.2 Hz, so the
    // button re-arms on a timer rather than pretending the work is done.
    setTimeout(() => {
      el.enrol.disabled = false;
      el.enrol.textContent = "Re-enrol face";
    }, 5500);
  }
});

boot();
