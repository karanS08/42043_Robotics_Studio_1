"""Part B/C: empirical characterisation of the cube-detection noise, used to set R of the ObjectKF.

For many static viewing poses (camera-to-cube range 0.15 - 0.50 m, random cube pose) the full sensing chain
(noisy RGB-D -> segmentation -> back-projection -> FK(q_hat) -> hand-eye) is run repeatedly; the error against
simulator truth (allowed here: this is an evaluation script) is binned by range and variance = a + b * range^2 is fitted.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation as Rot

from sim import scene as S
from sim.env import Plant, PlantConfig
from core.kinematics import RobotModel
from core.estimation import JointEKF
from core.sensing import Calibration, CubeDetector, wrap_quarter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect(n_poses=70, samples=15, seed=0):
    rng = np.random.default_rng(seed)
    robot = RobotModel(); calib = Calibration.load()
    det = CubeDetector(robot, calib)
    _, R_home = robot.fk(S.HOME_Q)
    rows = []
    for i in range(n_poses):
        cube_xy = (rng.uniform(0.18, 0.27), rng.uniform(-0.2, 0.2))
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        plant = Plant(PlantConfig(cube_xy=cube_xy, cube_yaw=yaw, seed=int(rng.integers(1 << 30))))
        z = rng.uniform(0.08, 0.30)
        tgt = np.array([cube_xy[0] + rng.uniform(-0.04, 0.04), cube_xy[1] + rng.uniform(-0.04, 0.04), z])
        Rd = Rot.from_euler("z", rng.uniform(-0.5, 0.5)).as_matrix() @ R_home
        q, ok = robot.ik(S.HOME_Q, tgt, Rd, iters=150)
        if not ok:
            plant.close(); continue
        u = robot.servo_equilibrium_u(q)
        ekf = JointEKF(robot, S.HOME_Q, accel_sigma=2.0)
        ekf.x[:6] = S.HOME_Q
        # move there smoothly then hold
        for k in range(400):
            s = min(1, k / 250); sm = s * s * (3 - 2 * s)
            uk = robot.servo_equilibrium_u(S.HOME_Q + (q - S.HOME_Q) * sm)
            obs = plant.step(uk); ekf.predict(uk); ekf.update(obs.q_meas)
        n = 0
        while n < samples:
            obs = plant.step(u); ekf.predict(u); ekf.update(obs.q_meas)
            if obs.rgb is None:
                continue
            m = det.detect(obs.rgb, obs.depth, ekf.q)
            n += 1
            if m is None:
                continue
            tp, ty = plant.truth.cube_pos, plant.truth.cube_yaw
            rows.append([m.range, *(m.pos - tp), wrap_quarter(m.yaw - ty), m.n_pix, i])
        plant.close()
    return np.array(rows)


def fit(r, e2):
    A = np.stack([np.ones_like(r), r ** 2], axis=1)
    sol, _ = nnls(A, e2)
    return sol


def main():
    rows = collect()
    np.save(os.path.join(ROOT, "results", "noise_samples.npy"), rows)
    r = rows[:, 0]
    out = {}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for j, (name, key) in enumerate([("x", "xy"), ("y", "xy"), ("z", "z"), ("yaw", "yaw")]):
        e = rows[:, 1 + j]
        ax[j].scatter(r, e * (1000 if name != "yaw" else 57.3), s=4, alpha=0.4)
        ax[j].set_title(f"{name} error vs range"); ax[j].set_xlabel("range [m]")
        ax[j].set_ylabel("mm" if name != "yaw" else "deg"); ax[j].axhline(0, c="k", lw=0.5)
    # split every error into a per-pose mean (bias: calibration + view geometry) and a within-pose residual (white noise)
    pose = rows[:, 6].astype(int)
    E = rows[:, 1:5]
    means = np.array([E[pose == k].mean(0) for k in np.unique(pose)])
    white = np.concatenate([E[pose == k] - E[pose == k].mean(0) for k in np.unique(pose)])
    r_w = np.concatenate([r[pose == k] for k in np.unique(pose)])
    a, b = fit(np.concatenate([r_w, r_w]), np.concatenate([white[:, 0] ** 2, white[:, 1] ** 2])); out.update(xy_a=float(a), xy_b=float(b))
    a, b = fit(r_w, white[:, 2] ** 2); out.update(z_a=float(a), z_b=float(b))
    a, b = fit(r_w, white[:, 3] ** 2); out.update(yaw_a=float(a), yaw_b=float(b))
    out.update(bias_xy=float((means[:, :2] ** 2).mean()), bias_z=float((means[:, 2] ** 2).mean()),
               bias_yaw=float((means[:, 3] ** 2).mean()))
    print("white std [mm] x,y,z:", (white[:, :3].std(0) * 1000).round(3), " yaw [deg]:", np.degrees(white[:, 3].std()).round(3))
    print("bias  rms [mm] x,y,z:", (np.sqrt((means[:, :3] ** 2).mean(0)) * 1000).round(3),
          " yaw [deg]:", np.degrees(np.sqrt((means[:, 3] ** 2).mean())).round(3))
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results", "noise_characterisation.png"), dpi=130)
    out["source"] = "simulation"        # hw_characterise_noise.py writes "hardware"; hw_check.py warns on this
    json.dump(out, open(os.path.join(ROOT, "configs", "noise_model.json"), "w"), indent=1)
    print("samples:", len(rows))
    print("bias [mm] x,y,z:", (rows[:, 1:4].mean(0) * 1000).round(3), " yaw [deg]:", np.degrees(rows[:, 4].mean()).round(3))
    print("std  [mm] x,y,z:", (rows[:, 1:4].std(0) * 1000).round(3), " yaw [deg]:", np.degrees(rows[:, 4].std()).round(3))
    print("fit:", out)


if __name__ == "__main__":
    main()
