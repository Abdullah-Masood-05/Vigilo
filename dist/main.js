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
  noFace: document.getElementById("flag-noface"),
  multiFace: document.getElementById("flag-multiface"),
  object: document.getElementById("flag-object"),
  degraded: document.getElementById("flag-degraded"),
};

let sawDegradedEvent = false;
let viewBox = "";

// Events are edge-triggered and low-rate — the opposite of the polled
// snapshot. Silent until fusion lands at step 8.
listen("detection:event", (event) => {
  console.log("detection:event", event.payload);
  if (event.payload && String(event.payload.event).includes("degraded")) {
    sawDegradedEvent = true;
  }
});

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

  if (snap.error) {
    el.error.hidden = false;
    el.error.textContent = snap.error;
  } else {
    el.error.hidden = true;
  }

  drawOverlay(snap);
  drawHud(snap);
  drawFlags(snap);
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

  // Render path is here and ready; nothing produces objects until step 7.
  for (const obj of s.objects || []) {
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
  const objects = (snap.signals?.objects || []).length;

  el.hud.textContent = [
    `${snap.source}`,
    `cap ${fmt(st.capture_fps)} fps   det ${fmt(st.detect_fps)} fps   skipped ${st.frames_skipped ?? 0}`,
    `detect  p50 ${ms(st.detect_p50_us ?? 0)} ms   p95 ${ms(st.detect_p95_us ?? 0)} ms`,
    `preview p50 ${ms(snap.preview_p50_us ?? 0)} ms`,
    `faces ${faces}   objects ${objects}   seq ${snap.seq}`,
    poseLine(snap.signals?.head_pose),
    gazeLine(snap.signals?.gaze),
    // Permanent, not a probe: a viewport wider than the window silently crops
    // the frame and pushes right-anchored UI off-screen. An instrument should
    // report the geometry it is drawing into.
    `view ${window.innerWidth}x${window.innerHeight}  dpr ${window.devicePixelRatio}`,
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

function poseLine(pose) {
  if (!pose) return "pose    —";
  const f = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);
  return `pose    yaw ${f(pose.yaw_deg)}  pitch ${f(pose.pitch_deg)}  roll ${f(pose.roll_deg)}`;
}

function fmt(v) {
  return (v ?? 0).toFixed(1);
}

// Instantaneous signal state only. No timers, no thresholds — that is exactly
// what fusion is for, and duplicating it here would be a second source of
// truth. These pills get replaced by Event-driven ones at step 8.
function drawFlags(snap) {
  const faces = (snap.signals?.faces || []).length;
  const objects = (snap.signals?.objects || []).length;
  const degraded = sawDegradedEvent || (snap.degraded || []).length > 0;

  el.noFace.classList.toggle("on", snap.seq > 0 && faces === 0);
  el.multiFace.classList.toggle("on", faces >= 2);
  el.object.classList.toggle("on", objects > 0);
  el.degraded.classList.toggle("on", degraded);
}

boot();
