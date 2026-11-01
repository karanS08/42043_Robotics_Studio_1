"""Stage 3b: measure the REAL camera's detection noise, so the object filter's R describes this sensor.

No robot motion: the arm is left wherever you put it, the object is left still, and the full detection chain
is run repeatedly. Move the arm by hand (freedrive) between poses to cover a range of distances, exactly as
the simulated version sweeps viewing poses.

    python scripts/hw_characterise_noise.py --ip 192.168.1.102 --poses 8 --samples 30

Splits the error into a white part (frame to frame at a fixed pose) and a bias part (varies with viewpoint),
fits variance = a + b*range^2 for each, and writes configs/noise_model.json with source="hardware".
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from scipy.optimize import nnls

from hardware.config import HardwareConfig
from core.kinematics import RobotModel
from core.sensing import Calibration, CubeDetector, wrap_quarter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "noise_model.json")


def fit(r, e2):
    A = np.stack([np.ones_like(r), r ** 2], axis=1)
    sol, _ = nnls(A, e2)
    return float(sol[0]), float(sol[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip")
    ap.add_argument("--poses", type=int, default=8)
    ap.add_argument("--samples", type=int, default=30)
    a = ap.parse_args()
    cfg = HardwareConfig.load(ip=a.ip)
    calib = Calibration.load()
    robot = RobotModel()
    det = CubeDetector(robot, calib)

    from hardware.devices import URArm, make_camera
    arm = URArm(cfg).connect(control=False)
    cam = make_camera(cfg).connect()

    print("Leave the object still. Move the arm by hand to a new viewing pose for each round.\n"
          "The object must stay in view and un-clipped. 'q' to finish early.\n")
    per_pose = []
    for i in range(a.poses):
        if input(f"pose {i+1}/{a.poses} - press ENTER when steady > ").strip().lower() == "q":
            break
        rows = []
        for _ in range(a.samples):
            rgb, depth = cam.read()
            q = arm.q()
            m = det.detect(rgb, depth, q)
            if m is not None:
                rows.append([m.range, *m.pos, m.yaw])
        if len(rows) < max(5, a.samples // 3):
            print(f"  only {len(rows)}/{a.samples} detections - check lighting, colour threshold, framing")
            continue
        rows = np.array(rows)
        per_pose.append(rows)
        print(f"  {len(rows)} detections, range {rows[:,0].mean():.3f} m, "
              f"spread {1e3*rows[:,1:4].std(0).round(4)} mm")
    cam.close(); arm.close()
    if len(per_pose) < 3:
        print("need at least 3 usable poses"); return 1

    # white = deviation within a pose; bias = how much each pose's mean differs from the overall mean
    white, r_w, means = [], [], []
    for rows in per_pose:
        mu = rows[:, 1:].mean(0)
        white.append(rows[:, 1:] - mu); r_w.append(rows[:, 0]); means.append(mu)
    white = np.vstack(white); r_w = np.concatenate(r_w); means = np.array(means)
    means = means - means.mean(0)
    out = {}
    aa, bb = fit(np.concatenate([r_w, r_w]), np.concatenate([white[:, 0] ** 2, white[:, 1] ** 2]))
    out.update(xy_a=aa, xy_b=bb)
    out.update(dict(zip(("z_a", "z_b"), fit(r_w, white[:, 2] ** 2))))
    yaw_w = np.array([wrap_quarter(x) for x in white[:, 3]])
    out.update(dict(zip(("yaw_a", "yaw_b"), fit(r_w, yaw_w ** 2))))
    out.update(bias_xy=float((means[:, :2] ** 2).mean()), bias_z=float((means[:, 2] ** 2).mean()),
               bias_yaw=float((means[:, 3] ** 2).mean()), source="hardware", n_poses=len(per_pose))
    print(f"\nwhite std [mm] x,y,z: {(1e3*white[:, :3].std(0)).round(3)}  yaw {np.degrees(yaw_w.std()):.2f} deg")
    print(f"bias  rms [mm] x,y,z: {(1e3*np.sqrt((means[:, :3]**2).mean(0))).round(3)}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
