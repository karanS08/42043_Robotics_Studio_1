"""Parts C/D: tuning studies (each point = N randomised pick trials).

  ekf_q      : joint-EKF process noise (accel_sigma) under nominal / payload / force -> NEES, NIS, velocity RMSE
  ekf_r      : encoder R x {0.01 .. 100}                                           -> NEES, velocity RMSE, control chatter, tracking
  obj_r      : camera R x {0.01 .. 100} for the object KF (noise x4 plant)          -> object NEES, estimate error at grasp
  lqr        : LQR weights (q_max, u_max) + servo-only baseline                      -> tracking RMSE vs effort vs chatter
Writes results/tuning_<study>.json.
"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
from multiprocessing import get_context
from eval.runner import TrialConfig, run_trial
from core.control import ControllerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["success", "grasp_err_norm", "obj_xy_err_at_grasp", "track_rmse", "tcp_track_rmse", "q_est_rmse", "qd_est_rmse",
        "nees_joint_mean", "nis_joint_mean", "obj_nees_mean", "nis_obj_mean", "effort_rms", "u_chatter", "time",
        "obj_vel_rmse", "obj_vel_err_at_grasp"]


def grid():
    g = []
    for plant_name, plant in (("nominal", {}), ("payload_0.5kg", dict(payload_kg=0.5)), ("force_25N", dict(force_N=25.0))):
        for a in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0):
            g.append(("ekf_q", dict(plant=plant_name, accel_sigma=a), plant, dict(accel_sigma=a)))
    for rs in (0.01, 0.1, 1.0, 10.0, 100.0):
        g.append(("ekf_r", dict(enc_r_scale=rs), {}, dict(enc_r_scale=rs)))
    for rs in (0.01, 0.1, 1.0, 10.0, 100.0):
        g.append(("obj_r", dict(obj_r_scale=rs), dict(noise_scale=4.0), dict(obj_r_scale=rs, assumed_noise_scale=4.0)))
    for sa in (0.02, 0.05, 0.1, 0.2, 0.35, 0.8):
        for scen_name, scen in (("static+dropout", dict(dropout_after=2.5)), ("moving", dict(speed=0.04)),
                                ("moving+dropout", dict(speed=0.04, dropout_after=3.0)),
                                ("oscillating", dict(object_motion=("sine", 0.10, 0.08, 1)))):
            g.append(("obj_q", dict(obj_accel_sigma=sa, scenario=scen_name), scen, dict(obj_accel_sigma=sa)))
    for sp in (0.0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
        g.append(("speed", dict(speed=sp, model="cv"), dict(speed=sp), {}))
        g.append(("speed", dict(speed=sp, model="static"), dict(speed=sp), dict(obj_model="static")))
    g.append(("lqr", dict(label="servo only", q_max=None, u_max=None), {}, dict(use_lqr=False)))
    for qm in (0.02, 0.005, 0.001):
        for um in (0.1, 0.02, 0.004):
            g.append(("lqr", dict(label=f"q{qm}/u{um}", q_max=qm, u_max=um), {}, dict(q_max=qm, u_max=um)))
    return g


def _work(args):
    idx, seed, plant, ctrl = args
    try:
        r = run_trial(TrialConfig(seed=seed, ctrl=ControllerConfig(**ctrl), **plant))
        return idx, {k: r[k] for k in KEYS}
    except Exception as e:
        return idx, dict(success=False, error=repr(e))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    only = sys.argv[2:]
    g = [x for x in grid() if not only or x[0] in only]
    jobs = [(i, s, x[2], x[3]) for i, x in enumerate(g) for s in range(n)]
    t0 = time.time()
    # RAM guard: ~0.5 GB per worker; default 4 workers, recycle workers to avoid slow leaks. Run via scripts/capped.sh.
    n_workers = int(os.environ.get("UR3_WORKERS", "4"))
    with get_context("spawn").Pool(n_workers, maxtasksperchild=40) as pool:
        res = pool.map(_work, jobs, chunksize=2)
    print(f"{len(jobs)} trials in {time.time() - t0:.0f} s")
    out = {}
    for i, x in enumerate(g):
        rs = [r for j, r in res if j == i]
        row = dict(x[1]); row["n"] = len(rs); row["success_rate"] = float(np.mean([r["success"] for r in rs]))
        row["errors"] = int(sum("error" in r for r in rs))
        for k in KEYS[1:]:
            v = np.array([r.get(k, np.nan) for r in rs], float)
            row[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        out.setdefault(x[0], []).append(row)
    for study, rows in out.items():
        json.dump(rows, open(os.path.join(ROOT, "results", f"tuning_{study}.json"), "w"), indent=1)
        print("==", study)
        for r in rows:
            print("  ", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                         if k in ("plant", "accel_sigma", "obj_accel_sigma", "scenario", "enc_r_scale", "obj_r_scale", "speed", "model", "label", "success_rate", "nees_joint_mean",
                                  "nis_joint_mean", "qd_est_rmse", "obj_nees_mean", "obj_xy_err_at_grasp", "track_rmse",
                                  "tcp_track_rmse", "effort_rms", "u_chatter", "obj_vel_rmse", "errors")})


if __name__ == "__main__":
    main()
