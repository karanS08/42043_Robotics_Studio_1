"""Part B: camera intrinsic + hand-eye calibration from a checkerboard seen by the wrist camera.

Procedure: move the arm to N randomised views of a 7x6-inner-corner board, average the encoder readings while
static (this is what a real calibration would do), detect corners (sub-pixel), then
  1. cv2.calibrateCamera       -> K, per-view board pose T_cam_board, reprojection error
  2. cv2.calibrateHandEye      -> T_tool_cam (camera pose in the tool frame), using FK(q_mean) as T_base_tool
The estimates are compared with the simulator truth (printed + saved) but only the ESTIMATES go to
configs/calibration.json, which is what the pipeline consumes.
"""
import os, sys, json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
from sim import scene as S
from sim.env import Plant, PlantConfig
from core.kinematics import RobotModel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "calibration.json")
PATTERN = (7, 6)
SQ = 0.025
N_VIEWS = 40
PP = ((S.IMG_W - 1) / 2.0, (S.IMG_H - 1) / 2.0)   # pixel-centre convention (MuJoCo image centre)
BOARD_CENTRE = (0.22, 0.0)


def main(seed=0, n_views=N_VIEWS):
    rng = np.random.default_rng(seed)
    robot = RobotModel()
    plant = Plant(PlantConfig(board=True, cube_xy=BOARD_CENTRE, seed=seed))
    _, R_home = robot.fk(S.HOME_Q)
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    # inner corners: scene board squares are indexed (a,b); corner (i,j) sits at board-frame (i+1)*SQ.. offset below
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQ

    imgpts, objpts, R_t2b, t_t2b, used = [], [], [], [], 0
    q_prev = S.HOME_Q.copy()
    tries = 0
    while used < n_views and tries < 6 * n_views:
        tries += 1
        tgt = np.array([BOARD_CENTRE[0] + rng.uniform(-0.03, 0.03), BOARD_CENTRE[1] + rng.uniform(-0.04, 0.04),
                        rng.uniform(0.16, 0.26)])
        yaw = np.radians(rng.uniform(-35, 35))
        tilt = Rot.from_rotvec(np.radians(rng.uniform(-30, 30, 2)).tolist() + [0.0])
        Rd = Rot.from_euler("z", yaw).as_matrix() @ tilt.as_matrix() @ R_home
        q, ok = robot.ik(q_prev, tgt, Rd, iters=150)
        if not ok:
            continue
        u = robot.servo_equilibrium_u(q)
        for _ in range(150):
            plant.step(u)
        qs, frame = [], None
        for _ in range(12):                      # collect encoder samples until a camera frame arrives
            o = plant.step(u)
            qs.append(o.q_meas)
            if o.rgb is not None:
                frame = o.rgb
        while frame is None:
            o = plant.step(u); qs.append(o.q_meas)
            frame = o.rgb
        for _ in range(40):
            qs.append(plant.step(u).q_meas)
        q_bar = np.mean(qs, axis=0)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        imgpts.append(corners); objpts.append(objp)
        p_tool, R_tool = robot.fk_tool(q_bar)
        R_t2b.append(R_tool); t_t2b.append(p_tool.reshape(3, 1))
        used += 1
        q_prev = q
    plant.close()
    print(f"views used: {used} / tries {tries}")

    # Pinhole camera with square pixels and principal point at the image centre (standard sensor assumption):
    # only the focal length is estimated; distortion is fixed to zero.
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS)
    K0 = np.array([[200.0, 0, PP[0]], [0, 200.0, PP[1]], [0, 0, 1]])
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, (S.IMG_W, S.IMG_H), K0, None, flags=flags)
    print("reprojection RMS [px]:", rms)

    R_b2c = [cv2.Rodrigues(r)[0] for r in rvecs]
    R_c2t, t_c2t = cv2.calibrateHandEye(R_t2b, t_t2b, R_b2c, tvecs, method=cv2.CALIB_HAND_EYE_PARK)
    T = np.eye(4); T[:3, :3] = R_c2t; T[:3, 3] = t_c2t.ravel()

    K_true, T_true = S.intrinsics(), S.true_T_tool_cam_opencv()
    dT = np.linalg.inv(T_true) @ T
    res = dict(
        source="simulation",
        n_views=used, reproj_rms_px=float(rms),
        K=K.tolist(), T_tool_cam=T.tolist(),
        K_err_px=dict(fx=float(K[0, 0] - K_true[0, 0]), fy=float(K[1, 1] - K_true[1, 1]),
                      cx=float(K[0, 2] - K_true[0, 2]), cy=float(K[1, 2] - K_true[1, 2])),
        handeye_trans_err_mm=float(np.linalg.norm(dT[:3, 3]) * 1000),
        handeye_rot_err_deg=float(np.degrees(np.linalg.norm(Rot.from_matrix(dT[:3, :3]).as_rotvec()))),
        dist=dist.ravel().tolist(),
    )
    print(json.dumps({k: v for k, v in res.items() if k not in ("K", "T_tool_cam")}, indent=1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
