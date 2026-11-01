"""Stage 3: camera intrinsics and hand-eye calibration on the REAL camera and robot.

Same estimators as the simulation (cv2.calibrateCamera, cv2.calibrateHandEye/Park) so the two are directly
comparable; only the source of the images and poses differs.

Two modes:

  --manual   (default, no robot motion) You move the arm by hand in freedrive to each pose and press ENTER.
             Safest, and works before the servo loop is trusted.
  --auto     The robot moves itself through poses around the board. Requires scripts/hw_check.py to pass.

    python scripts/hw_calibrate.py --ip 192.168.1.102 --views 20
    python scripts/hw_calibrate.py --ip 192.168.1.102 --auto --views 25

Put a 7x6-inner-corner checkerboard of known square size flat on the table inside the workspace.
Writes configs/calibration.json with source="hardware"; the simulation file is backed up first.
"""
import argparse, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

from hardware.config import HardwareConfig
from core.kinematics import RobotModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "calibration.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip")
    ap.add_argument("--pattern", default="7x6", help="inner corners, cols x rows")
    ap.add_argument("--square", type=float, default=0.025, help="square size [m]")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--auto", action="store_true", help="robot moves itself between views")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    cfg = HardwareConfig.load(ip=a.ip)
    cols, rows = (int(x) for x in a.pattern.lower().split("x"))
    pattern = (cols, rows)
    if cols == rows:
        print("WARNING: a square pattern is 180-degree ambiguous; use an asymmetric one such as 7x6.")

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * a.square

    from hardware.devices import URArm, make_camera
    arm = URArm(cfg).connect(control=a.auto)
    cam = make_camera(cfg).connect()
    robot = RobotModel()
    if a.auto:
        print("\n*** THE ROBOT WILL MOVE between calibration views. ***")
        if not a.yes and input("    type 'move' to continue: ").strip() != "move":
            return 1
        arm.move_j(cfg.home_q); time.sleep(0.5)
    else:
        print("\nManual mode: put the robot in freedrive, point the wrist camera at the board from a new\n"
              "angle each time, hold still, then press ENTER. Vary tilt and distance; 'q' to finish early.")

    imgpts, objpts, R_t2b, t_t2b = [], [], [], []
    rng = np.random.default_rng(0)
    q_home = np.array(cfg.home_q)
    while len(imgpts) < a.views:
        if a.auto:
            tgt = q_home + rng.uniform(-0.25, 0.25, 6) * np.array([1, .5, .5, 1, 1, 1.5])
            try:
                arm.move_j(tgt)
            except Exception as e:
                print("  move failed:", e); continue
            time.sleep(0.6)
        else:
            if input(f"view {len(imgpts)+1}/{a.views} > ").strip().lower() == "q":
                break
        qs = [arm.q() for _ in range(20)]
        rgb, _ = cam.read()
        q_bar = np.mean(qs, axis=0)
        if float(np.max(np.std(qs, axis=0))) > 2e-3:
            print("  robot was still moving, skipping"); continue
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print("  board not found"); continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        imgpts.append(corners); objpts.append(objp)
        p_tool, R_tool = robot.fk_tool(q_bar)
        R_t2b.append(R_tool); t_t2b.append(p_tool.reshape(3, 1))
        print(f"  captured {len(imgpts)}/{a.views}")

    cam.close(); arm.close()
    if len(imgpts) < 8:
        print(f"only {len(imgpts)} usable views; need at least 8."); return 1

    h, w = gray.shape
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, (w, h), None, None, flags=flags)
    R_b2c = [cv2.Rodrigues(r)[0] for r in rvecs]
    R_c2t, t_c2t = cv2.calibrateHandEye(R_t2b, t_t2b, R_b2c, tvecs, method=cv2.CALIB_HAND_EYE_PARK)
    T = np.eye(4); T[:3, :3] = R_c2t; T[:3, 3] = t_c2t.ravel()

    # consistency check: the board pose in the base frame must agree across views
    base_pts = []
    for Rt, tt, Rc, tc in zip(R_t2b, t_t2b, R_b2c, tvecs):
        Tt = np.eye(4); Tt[:3, :3] = Rt; Tt[:3, 3] = tt.ravel()
        Tc = np.eye(4); Tc[:3, :3] = Rc; Tc[:3, 3] = tc.ravel()
        base_pts.append((Tt @ T @ Tc)[:3, 3])
    spread = float(np.std(np.array(base_pts), axis=0).max())

    res = dict(source="hardware", n_views=len(imgpts), reproj_rms_px=float(rms),
               K=K.tolist(), dist=dist.ravel().tolist(), T_tool_cam=T.tolist(),
               board_spread_mm=1e3 * spread, pattern=a.pattern, square_m=a.square,
               image_size=[int(w), int(h)], captured=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nreprojection RMS   {rms:.3f} px   (good: < 0.5)")
    print(f"hand-eye spread    {1e3*spread:.1f} mm   (good: < 5; this is the real accuracy limit of the pick)")
    print(f"T_tool_cam translation {np.round(T[:3,3], 4)} m")
    if os.path.exists(OUT):
        shutil.copy(OUT, OUT.replace(".json", ".sim_backup.json"))
        print(f"backed up the previous calibration to {os.path.basename(OUT).replace('.json', '.sim_backup.json')}")
    import json
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    print("\nNext: scripts/hw_characterise_noise.py, then scripts/hw_run.py --ip ... (dry run first).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
