"""Live demo driver for the Thursday simulation demonstration.

Opens the interactive MuJoCo viewer (needs a display) and runs the full pipeline in real time, printing the
task phase and the live estimates. Ground truth is printed only in the evaluation column, never used by the
controller.

Examples
--------
  python scripts/demo.py                          # static cube
  python scripts/demo.py --speed 0.04             # moving object: the tracker earns its place
  python scripts/demo.py --speed 0.04 --static-model   # same, with the static-object filter: it fails
  python scripts/demo.py --speed 0.04 --dropout 3.0    # camera dies mid-approach, estimate coasts
  python scripts/demo.py --oscillate              # object slides back and forth
  python scripts/demo.py --speed 0.04 --record out.mp4 # no viewer, write a video instead
"""
import argparse, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHASES = ["SCAN", "PREGRASP", "DESCEND", "GRASP", "LIFT", "DONE", "FAILED"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--speed", type=float, default=0.0, help="object speed [m/s], 0 = static")
    ap.add_argument("--oscillate", action="store_true", help="object slides back and forth instead")
    ap.add_argument("--noise", type=float, default=1.0, help="sensor noise multiplier")
    ap.add_argument("--dropout", type=float, default=None, help="kill the camera at this time [s]")
    ap.add_argument("--force", type=float, default=0.0, help="disturbance force on the wrist [N]")
    ap.add_argument("--static-model", action="store_true", help="use the static-object filter (ablation)")
    ap.add_argument("--no-kf", action="store_true", help="use raw detections, no object filter (ablation)")
    ap.add_argument("--handeye", type=float, default=0.0, help="injected hand-eye error [mm]")
    ap.add_argument("--record", type=str, default=None, help="write an mp4 instead of opening the viewer")
    ap.add_argument("--realtime", action="store_true", help="throttle the viewer to wall-clock speed")
    a = ap.parse_args()

    if a.record:
        os.environ.setdefault("MUJOCO_GL", "egl")
    from eval.runner import TrialConfig, run_trial, sample_plant_cfg
    from core.control import ControllerConfig, Phase
    from core.sensing import Calibration
    from sim.env import Plant, CTRL_DT
    from core.control import PickController

    ctrl = ControllerConfig(obj_model="static" if a.static_model else "cv", use_object_kf=not a.no_kf)
    tc = TrialConfig(seed=a.seed, speed=a.speed or None, noise_scale=a.noise, dropout_after=a.dropout,
                     force_N=a.force, ctrl=ctrl,
                     object_motion=("sine", 0.10, 0.08, 1) if a.oscillate else None,
                     handeye_err=(a.handeye / 1000.0, a.handeye / 10.0))

    if a.record:
        tc.keep_log = True; tc.frames = True
        from scripts.make_video import make_clip  # noqa
        print("recording ...")
        r = run_trial(tc)
        print(f"success={r['success']}  grasp error={1e3*r['grasp_err_norm']:.2f} mm")
        return

    import mujoco.viewer
    pc = sample_plant_cfg(tc)
    plant = Plant(pc)
    calib = Calibration.load()
    if a.handeye:
        calib = calib.perturbed(a.handeye / 1000.0, a.handeye / 10.0, np.random.default_rng(a.seed + 7))
    ctl = PickController(ctrl, calib)
    obs = plant._last_obs
    print(f"cube at ({pc.cube_xy[0]:.3f}, {pc.cube_xy[1]:.3f}) yaw {np.degrees(pc.cube_yaw):+.0f} deg"
          f"   motion={pc.object_motion}   noise x{pc.noise_scale:g}")
    print(f"{'t':>6} {'phase':>9} {'obj est (x,y,z) mm':>26} {'|err| mm':>9} {'v est mm/s':>22} {'|v err|':>8}")
    last = Phase.SCAN
    t0 = time.time()
    with mujoco.viewer.launch_passive(plant.model, plant.data) as v:
        for k in range(int((tc.ctrl.timeout + 1.0) / CTRL_DT)):
            u, grip, info = ctl.step(obs)
            obs = plant.step(u, grip)
            if k % 2 == 0:
                v.sync()
            if a.realtime:
                dt = t0 + k * CTRL_DT - time.time()
                if dt > 0:
                    time.sleep(dt)
            ph = Phase(info["phase"])
            if ph != last or k % 50 == 0:
                o = info["obj"]
                if o is not None:
                    true = np.array([*plant.truth.cube_pos, plant.truth.cube_yaw])
                    vt = np.array([*plant.truth.cube_vel, 0.0])
                    e = 1e3 * np.linalg.norm(o[:3] - true[:3])
                    ve = 1e3 * np.linalg.norm(info["obj_vel"][:3] - vt[:3])
                    print(f"{info['t']:6.2f} {PHASES[ph]:>9} "
                          f"{np.array2string(1e3*o[:3], precision=1, floatmode='fixed'):>26} {e:9.2f} "
                          f"{np.array2string(1e3*info['obj_vel'][:3], precision=1, floatmode='fixed'):>22} {ve:8.2f}")
                else:
                    print(f"{info['t']:6.2f} {PHASES[ph]:>9}   (no detection yet)")
                last = ph
            if ph in (Phase.DONE, Phase.FAILED):
                break
        held = plant.truth.holding and plant.truth.cube_pos[2] > 0.12
        print(f"\nresult: phase={PHASES[ctl.phase]}  lifted={held}  "
              f"object updates={ctl.obj.n_updates} rejected={ctl.obj.n_rejected}")
        for _ in range(150):
            plant.step(u, grip); v.sync(); time.sleep(0.01)


if __name__ == "__main__":
    main()
