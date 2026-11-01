"""Stage 4: run the pick on the real UR3e.

    python scripts/hw_run.py --ip 192.168.1.102              # dry run: estimates only, NO motion
    python scripts/hw_run.py --ip 192.168.1.102 --move       # the robot moves

Dry run is the default on purpose. It executes the entire pipeline - camera, detector, both filters, the
controller - and prints the commands it WOULD send, without connecting the control interface. Use it to
confirm the object is detected and the estimates are sane before anything moves.

Logs every step to results/hw_<timestamp>.npz so a hardware run can be compared against simulation with the
same metrics (scripts/hw_compare.py).
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from hardware.config import HardwareConfig
from hardware.safety import SafetyAbort
from core.control import PickController, ControllerConfig, Phase, CTRL_DT
from core.kinematics import RobotModel
from core.sensing import Calibration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASES = ["SCAN", "PREGRASP", "DESCEND", "GRASP", "LIFT", "DONE", "FAILED"]


def build_controller(cfg: HardwareConfig, args) -> PickController:
    ff, kp, kd = "none", None, None
    if cfg.servo_kp and cfg.servo_kd and not args.no_gains:
        ff, kp, kd = "gains", tuple(cfg.servo_kp), tuple(cfg.servo_kd)
    cc = ControllerConfig(
        feedforward=ff, servo_kp=kp, servo_kd=kd,
        obj_model="cv",
        ws_x=cfg.ws_x, ws_y=cfg.ws_y, ws_z=cfg.ws_z,
        timeout=args.timeout,
        v_max=args.v_max,
    )
    print(f"controller: feedforward={ff}" + (f" (identified gains kp={np.round(kp,0)})" if kp else
          "  <- servo gains NOT identified; run scripts/hw_identify_servo.py for a platform-specific model"))
    return PickController(cc, Calibration.load())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip")
    ap.add_argument("--move", action="store_true", help="actually move the robot (default is a dry run)")
    ap.add_argument("--no-home", action="store_true", help="skip the initial moveJ to the home pose")
    ap.add_argument("--no-gains", action="store_true", help="ignore identified servo gains")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--v-max", type=float, default=0.08, help="Cartesian speed limit [m/s] (sim default 0.25)")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    cfg = HardwareConfig.load(ip=a.ip)

    calib_path = os.path.join(ROOT, "configs", "calibration.json")
    src = json.load(open(calib_path)).get("source", "unknown") if os.path.exists(calib_path) else None
    if src != "hardware":
        print(f"WARNING: configs/calibration.json has source={src!r}, not 'hardware'.\n"
              "         The intrinsics and hand-eye transform belong to the simulated camera; on the real\n"
              "         robot the frame chain will be wrong. Run scripts/hw_calibrate.py first.\n")

    ctl = build_controller(cfg, a)
    rm = ctl.robot

    from hardware.plant import HardwarePlant
    plant = HardwarePlant(cfg, rm)

    if a.move:
        print(f"\n*** THE ROBOT WILL MOVE ***  {cfg.ip}")
        print(f"    TCP box x{cfg.ws_x} y{cfg.ws_y} z{cfg.ws_z} m, max {cfg.max_joint_speed} rad/s, "
              f"abort if {cfg.max_tracking_error} rad behind.")
        if not a.yes and input("    Area clear, e-stop in hand? type 'move' to continue: ").strip() != "move":
            print("aborted."); return 1
    else:
        print("\nDRY RUN - the control interface is not used and the robot will not move.")
        print("Add --move once the estimates look right.\n")

    log = {k: [] for k in ("t", "q", "u", "phase", "obj", "obj_vel", "obj_P", "tcp_cmd", "nis_joint")}
    t_start = time.time()
    try:
        plant.connect(control=bool(a.move))
        if a.move and not a.no_home:
            print("moving to home ..."); plant.go_home()
        obs = plant.reset()
        print(f"{'t':>6} {'phase':>9} {'object est (x,y,z) mm':>26} {'speed mm/s':>11} {'|u-q| mrad':>11}")
        last_print = -1.0
        while True:
            u, grip, info = ctl.step(obs)
            ph = Phase(info["phase"])
            log["t"].append(info["t"]); log["q"].append(obs.q_meas.copy()); log["u"].append(u.copy())
            log["phase"].append(int(ph)); log["nis_joint"].append(info["nis_joint"])
            log["obj"].append(info["obj"].copy() if info["obj"] is not None else np.full(4, np.nan))
            log["obj_vel"].append(info["obj_vel"].copy())
            log["obj_P"].append(np.diag(info["obj_P"]).copy() if info["obj_P"] is not None else np.full(4, np.nan))
            log["tcp_cmd"].append(info["p_ref"].copy())
            if info["t"] - last_print > 0.5 or ph in (Phase.DONE, Phase.FAILED):
                o = info["obj"]
                sp = 1e3 * float(np.linalg.norm(info["obj_vel"][:3]))
                res = 1e3 * float(np.max(np.abs(u - obs.q_meas)))
                print(f"{info['t']:6.2f} {PHASES[ph]:>9} "
                      + (f"{np.array2string(1e3*o[:3], precision=1, floatmode='fixed'):>26}" if o is not None
                         else f"{'(no detection)':>26}")
                      + f" {sp:11.1f} {res:11.2f}")
                last_print = info["t"]
            if ph in (Phase.DONE, Phase.FAILED):
                print(f"\nfinished: {PHASES[ph]}  updates={ctl.obj.n_updates} rejected={ctl.obj.n_rejected}")
                break
            obs = plant.step(u, grip) if a.move else _dry_step(plant, obs, info)
    except SafetyAbort as e:
        print(f"\nSAFETY ABORT: {e}")
    except KeyboardInterrupt:
        print("\ninterrupted by user")
    finally:
        plant.close()
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        out = os.path.join(ROOT, "results", f"hw_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        np.savez_compressed(out, **{k: np.array(v) for k, v in log.items()},
                            meta=json.dumps(dict(ip=cfg.ip, moved=bool(a.move), duration=time.time() - t_start)))
        print("log:", out)
    return 0


def _dry_step(plant, obs, info):
    """Advance the clock and re-read the sensors without commanding anything."""
    import time as _t
    _t.sleep(CTRL_DT)
    plant.k += 1
    plant._last_obs = plant._observe(_t.monotonic() - plant.t0)
    return plant._last_obs


if __name__ == "__main__":
    sys.exit(main())
