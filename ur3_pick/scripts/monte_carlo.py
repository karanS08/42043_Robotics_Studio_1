"""Part E: Monte Carlo evaluation over randomised cube poses and disturbance/noise conditions.

Usage: python scripts/monte_carlo.py [n_trials] [conditions...]
Writes results/mc_<condition>.json (one record per trial) and prints a summary table.
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # one BLAS thread per worker process
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
from multiprocessing import get_context
from eval.runner import TrialConfig, run_trial
from core.control import ControllerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONDITIONS = {
    "nominal":        dict(),
    "noise_x2":       dict(noise_scale=2.0),
    "noise_x4":       dict(noise_scale=4.0),
    "noise_x8":       dict(noise_scale=8.0),
    "cam_dropout":    dict(dropout_after=2.5),                 # camera dies after the scan: estimate coasts
    "force_10N":      dict(force_N=10.0),
    "force_25N":      dict(force_N=25.0),
    "payload_0.5kg":  dict(payload_kg=0.5),
    "handeye_5mm_1deg":  dict(handeye_err=(0.005, 1.0)),
    "handeye_10mm_2deg": dict(handeye_err=(0.010, 2.0)),
    "handeye_20mm_4deg": dict(handeye_err=(0.020, 4.0)),
    # ---- moving object (the tracking / visual-servoing scenario) ----
    "moving_0.02":    dict(speed=0.02),
    "moving_0.04":    dict(speed=0.04),
    "moving_0.06":    dict(speed=0.06),
    "oscillating":    dict(object_motion=("sine", 0.10, 0.08, 1)),
    "moving_noise_x4": dict(speed=0.04, noise_scale=4.0),
    "moving_dropout": dict(speed=0.04, dropout_after=3.0),
    "moving_force_25N": dict(speed=0.04, force_N=25.0),
    # ablations (same nominal plant)
    "abl_servo_only": dict(ctrl=dict(use_lqr=False)),
    "abl_no_objkf":   dict(ctrl=dict(use_object_kf=False)),
    "abl_no_objkf_noise_x4": dict(noise_scale=4.0, ctrl=dict(use_object_kf=False)),
    "abl_p90_height_noise_x4": dict(noise_scale=4.0, ctrl=dict(z_estimator="p90")),
    "abl_servo_only_force_25N": dict(force_N=25.0, ctrl=dict(use_lqr=False)),
    # ---- ablations that isolate what the tracker contributes on a MOVING object ----
    "abl_mov_static_model": dict(speed=0.04, ctrl=dict(obj_model="static")),
    "abl_mov_no_objkf":     dict(speed=0.04, ctrl=dict(use_object_kf=False)),
    "abl_mov_no_vel_ff":    dict(speed=0.04, ctrl=dict(track_object_velocity=False)),
    "abl_mov_no_lead":      dict(speed=0.04, ctrl=dict(lead_time=0.0)),
}


def make(cond: str, seed: int) -> TrialConfig:
    kw = dict(CONDITIONS[cond])
    ctrl = ControllerConfig(**kw.pop("ctrl", {}))
    return TrialConfig(seed=seed, ctrl=ctrl, **kw)


def _work(args):
    cond, seed = args
    try:
        r = run_trial(make(cond, seed))
    except Exception as e:      # a crashed trial is a failed trial, and is reported as such
        r = dict(seed=seed, success=False, error=repr(e))
    r["condition"] = cond
    return r


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    conds = sys.argv[2:] or list(CONDITIONS)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    jobs = [(c, s) for c in conds for s in range(n)]
    t0 = time.time()
    # RAM guard: ~0.5 GB per worker; default 4 workers, recycle workers to avoid slow leaks. Run via scripts/capped.sh.
    n_workers = int(os.environ.get("UR3_WORKERS", "4"))
    with get_context("spawn").Pool(n_workers, maxtasksperchild=40) as pool:
        out = pool.map(_work, jobs, chunksize=2)
    print(f"{len(jobs)} trials in {time.time() - t0:.0f} s")
    for c in conds:
        rs = [r for r in out if r["condition"] == c]
        json.dump(rs, open(os.path.join(ROOT, "results", f"mc_{c}.json"), "w"), indent=0, default=float)
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        g = np.array([r.get("grasp_err_norm", np.nan) for r in rs], float)
        print(f"{c:28s} success {k:3d}/{len(rs)} ({100*k/len(rs):5.1f} %, 95% CI {100*lo:4.1f}-{100*hi:5.1f})  "
              f"grasp err med {1000*np.nanmedian(g):5.2f} mm p95 {1000*np.nanpercentile(g,95):5.2f} mm  errors {sum('error' in r for r in rs)}")


if __name__ == "__main__":
    main()
