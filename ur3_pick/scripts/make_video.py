"""Demo video clips: external view + wrist camera + live estimate/uncertainty overlay.

Renders one clip per scenario to results/video/, each with a side panel showing
  - the phase of the pick
  - the object estimate and its +-2 sigma against the true pose (truth is evaluation-only data, drawn as a reference)
  - the joint-velocity estimate, which is never measured
Combine/narrate them to make the 5-minute submission video.
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

from eval.runner import TrialConfig, run_trial
from core.control import ControllerConfig, Phase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "video")
PHASES = ["SCAN", "PREGRASP", "DESCEND", "GRASP", "LIFT", "DONE", "FAILED"]
FPS = 25


def panel(A, k_log, cam_img, title, W=600, H=480):
    """Right-hand panel: wrist-camera view on top, live estimate traces below."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    axcam = fig.add_axes([0.055, 0.545, 0.40, 0.325])
    axcam.imshow(cam_img); axcam.set_xticks([]); axcam.set_yticks([])
    axcam.set_title("wrist camera (what the estimator sees)", fontsize=7)
    t = A["t"][:k_log + 1]
    tmax = max(1.0, A["t"][-1])
    ax1 = fig.add_axes([0.60, 0.775, 0.37, 0.145])
    ax2 = fig.add_axes([0.60, 0.565, 0.37, 0.145])
    ax3 = fig.add_axes([0.10, 0.115, 0.87, 0.31])
    oe, osig = A["obj_err"][:k_log + 1], A["obj_sig"][:k_log + 1]
    for i, c in ((0, "C0"), (1, "C1")):
        ax1.plot(t, oe[:, i] * 1e3, c, lw=1.0, label="xy"[i])
        ax1.fill_between(t, -2e3 * osig[:, i], 2e3 * osig[:, i], color=c, alpha=0.15)
    ax1.set_ylabel("obj err\n[mm]", fontsize=6.5); ax1.set_ylim(-12, 12)
    ax1.legend(fontsize=5.5, loc="upper right", ncol=2, handlelength=1)
    ax2.plot(t, np.degrees(oe[:, 3]), "C3", lw=1.0)
    ax2.fill_between(t, -2 * np.degrees(osig[:, 3]), 2 * np.degrees(osig[:, 3]), color="C3", alpha=0.15)
    ax2.set_ylabel("yaw err\n[deg]", fontsize=6.5); ax2.set_ylim(-12, 12)
    vt = A["obj_vel_true"][:k_log + 1]
    moving = np.isfinite(vt).any() and np.nanmax(np.abs(vt[:, :2])) > 1e-3
    if moving:
        for i, (lbl, c) in enumerate((("vx", "C0"), ("vy", "C1"))):
            ax3.plot(t, 1e3 * A["obj_vel_true"][:k_log + 1, i], c + "--", lw=1.4, label=f"true {lbl}")
            ax3.plot(t, 1e3 * A["obj_vel_hat"][:k_log + 1, i], c, lw=1.1, label=f"estimated {lbl}")
        ax3.set_ylabel("object velocity [mm/s]\n(never measured)", fontsize=6.5)
        ax3.set_ylim(-70, 70); ax3.legend(fontsize=6, loc="upper right", ncol=2)
    else:
        ev = A["qd_hat"][:k_log + 1, 1] - A["qd"][:k_log + 1, 1]
        sv = 2 * np.sqrt(A["P_v"][:k_log + 1, 1])
        ax3.plot(t, ev, "C2", lw=1.0, label="error")
        ax3.fill_between(t, -sv, sv, color="C2", alpha=0.18, label=r"$\pm 2\sigma$")
        ax3.set_ylabel("shoulder-lift velocity error [rad/s]\n(never measured)", fontsize=6.5)
        ax3.set_ylim(-0.25, 0.25); ax3.legend(fontsize=6, loc="upper right", ncol=2)
    ax3.set_xlabel("t [s]", fontsize=7)
    for a_, ttl in ((ax1, "object position estimate vs truth, $\\pm2\\sigma$"),
                    (ax2, "object yaw estimate vs truth"), (ax3, None)):
        a_.grid(alpha=0.3); a_.tick_params(labelsize=5.5); a_.set_xlim(0, tmax)
        if ttl:
            a_.set_title(ttl, fontsize=6.5, pad=2)
    ph = PHASES[int(A["phase"][k_log])]
    fig.text(0.055, 0.962, title, fontsize=9.5, weight="bold")
    fig.text(0.50, 0.50, f"t = {A['t'][k_log]:5.2f} s", fontsize=9, family="monospace")
    fig.text(0.50, 0.455, f"phase: {ph}", fontsize=9, weight="bold", color="#b91c1c", family="monospace")
    canvas = FigureCanvasAgg(fig); canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def make_clip(name, tc, title):
    r = run_trial(tc)
    A, frames = r["log"], r["frames"]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.mp4")
    H = 480
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8, ffmpeg_log_level="error") as w:
        for f in frames:
            k = int(round(f["t"] / 0.01))
            k = min(k, len(A["t"]) - 1)
            frame = np.concatenate([f["ext"], panel(A, k, f["cam"], title)], axis=1)
            w.append_data(frame)
    print(f"  {name}: success={r['success']} grasp_err={1e3*r['grasp_err_norm']:.2f} mm -> {path} ({len(frames)} frames)")
    return r


def main():
    which = sys.argv[1:] or ["nominal", "moving", "moving_static_model", "moving_dropout", "oscillating",
                             "noise_x4", "cam_dropout", "force_25N"]
    specs = dict(
        nominal=(TrialConfig(seed=2, keep_log=True, frames=True), "Nominal pick, static object"),
        moving=(TrialConfig(seed=2, speed=0.04, keep_log=True, frames=True),
                "Moving object, 40 mm/s: velocity is estimated, never measured"),
        moving_static_model=(TrialConfig(seed=2, speed=0.04, keep_log=True, frames=True,
                                         ctrl=ControllerConfig(obj_model="static")),
                             "Same object, static-model filter: it fails (0/50)"),
        moving_dropout=(TrialConfig(seed=2, speed=0.04, dropout_after=3.0, keep_log=True, frames=True),
                        "Moving object, camera lost at 3 s: the tracker coasts"),
        oscillating=(TrialConfig(seed=2, object_motion=("sine", 0.10, 0.08, 1), keep_log=True, frames=True),
                     "Oscillating object: constant-velocity model lags each reversal"),
        noise_x4=(TrialConfig(seed=2, noise_scale=4.0, keep_log=True, frames=True), "4x sensor noise"),
        cam_dropout=(TrialConfig(seed=2, dropout_after=2.5, keep_log=True, frames=True),
                     "Camera lost after scan (estimate coasts)"),
        force_25N=(TrialConfig(seed=2, force_N=25.0, keep_log=True, frames=True), "25 N disturbance on the wrist"),
    )
    summary = {}
    for k in which:
        tc, title = specs[k]
        r = make_clip(k, tc, title)
        summary[k] = {kk: r[kk] for kk in ("success", "grasp_err_norm", "time")}
    json.dump(summary, open(os.path.join(OUT, "clips.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
