"""Trial runner + metrics. This is the ONLY place where controller outputs and simulator ground truth meet.

Ground truth (plant.truth) is used here for logging/evaluation and is never passed to the controller:
the controller receives `Observation` objects only.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
from sim.env import Plant, PlantConfig, CTRL_DT
from core.control import PickController, ControllerConfig, Phase
from core.sensing import Calibration, wrap_quarter

WORKSPACE_X = (0.18, 0.27)
WORKSPACE_Y = (-0.20, 0.20)


@dataclass
class TrialConfig:
    seed: int = 0
    noise_scale: float = 1.0
    dropout_after: Optional[float] = None       # camera dies at this time [s] (None = never)
    dropout_window: Optional[Tuple[float, float]] = None
    force_N: float = 0.0                        # impulsive disturbance magnitude on the wrist
    payload_kg: float = 0.0                     # unmodelled payload
    handeye_err: Tuple[float, float] = (0.0, 0.0)   # (metres, degrees) injected calibration error
    object_motion: Optional[tuple] = None           # None = static; ("linear",vx,vy) or ("sine",amp,freq,axis)
    speed: Optional[float] = None                   # convenience: object speed [m/s], direction randomised
    ctrl: ControllerConfig = field(default_factory=ControllerConfig)
    keep_log: bool = False
    frames: bool = False                        # keep camera/external frames for the video


def sample_plant_cfg(tc: TrialConfig) -> PlantConfig:
    rng = np.random.default_rng(tc.seed)
    xy = (rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y))
    yaw = rng.uniform(-np.pi / 4, np.pi / 4)
    drops = []
    if tc.dropout_after is not None:
        drops.append((tc.dropout_after, 1e9))
    if tc.dropout_window is not None:
        drops.append(tc.dropout_window)
    dist = None
    if tc.force_N > 0:
        ang = rng.uniform(0, 2 * np.pi)
        t0 = rng.uniform(3.0, 7.0)
        dist = (t0, 0.3, (tc.force_N * np.cos(ang), tc.force_N * np.sin(ang), 0.0))
    motion = tc.object_motion
    if motion is None and tc.speed:
        # drive the object across the workspace: start on one side, travel toward the other
        sgn = 1.0 if xy[1] < 0 else -1.0
        motion = ("linear", 0.0, sgn * tc.speed)
    return PlantConfig(noise_scale=tc.noise_scale, cube_xy=xy, cube_yaw=yaw, camera_dropouts=tuple(drops),
                       disturbance=dist, payload_kg=tc.payload_kg, object_motion=motion, seed=tc.seed + 100003)


def run_trial(tc: TrialConfig) -> dict:
    pc = sample_plant_cfg(tc)
    plant = Plant(pc)
    calib = Calibration.load()
    if tc.handeye_err[0] or tc.handeye_err[1]:
        calib = calib.perturbed(tc.handeye_err[0], tc.handeye_err[1], np.random.default_rng(tc.seed + 7))
    ctl = PickController(tc.ctrl, calib)
    truth = plant.truth
    obs = plant._last_obs
    L = {k: [] for k in ("t", "phase", "q", "qd", "q_hat", "qd_hat", "q_ref", "qd_ref", "nees_joint", "nis_joint", "u", "tau",
                         "tcp", "tcp_hat", "p_ref", "obj_err", "obj_sig", "obj_nees", "obj_raw_err", "nis_obj", "P_q", "P_v",
                         "obj_vel_err", "obj_vel_hat", "obj_vel_true")}
    frames = []
    cube0, yaw0 = truth.cube_pos, truth.cube_yaw
    grasp_err = None
    obj_err_at_grasp = raw_err_at_grasp = None
    vel_err_at_grasp = None
    n_steps = int((tc.ctrl.timeout + 1.0) / CTRL_DT)
    done_k = None
    diverged = False
    for k in range(n_steps):
        if not (np.all(np.isfinite(truth.q)) and np.all(np.isfinite(truth.qd))):
            diverged = True             # the plant has gone non-finite: a failed trial, not a crashed script
            break
        u, grip, info = ctl.step(obs)
        if not np.all(np.isfinite(u)):
            diverged = True
            break
        # ---------------- evaluation only (ground truth) ----------------
        q, qd = truth.q, truth.qd
        e = np.concatenate([info["x_hat"][:6] - q, info["x_hat"][6:] - qd])
        L["t"].append(info["t"]); L["phase"].append(info["phase"])
        L["q"].append(q); L["qd"].append(qd); L["q_hat"].append(info["x_hat"][:6]); L["qd_hat"].append(info["x_hat"][6:])
        L["q_ref"].append(info["q_ref"]); L["qd_ref"].append(info["qd_ref"])
        L["nees_joint"].append(float(e @ np.linalg.solve(info["P_joint"], e))); L["nis_joint"].append(info["nis_joint"])
        L["P_q"].append(np.diag(info["P_joint"])[:6]); L["P_v"].append(np.diag(info["P_joint"])[6:])
        L["u"].append(info["u"]); L["tau"].append(truth.u_effort[:6])
        L["tcp"].append(truth.tcp_pos); L["tcp_hat"].append(info["tcp_hat"]); L["p_ref"].append(info["p_ref"])
        if info["obj"] is not None and not truth.holding and info["phase"] <= Phase.GRASP:
            oe = info["obj"] - np.array([*truth.cube_pos, truth.cube_yaw]); oe[3] = wrap_quarter(oe[3])
            L["obj_err"].append(oe); L["obj_sig"].append(np.sqrt(np.diag(info["obj_P"])))
            L["obj_nees"].append(float(oe @ np.linalg.solve(info["obj_P"], oe)) if info["new_meas"] else np.nan)
            re = info["obj_raw"] - np.array([*truth.cube_pos, truth.cube_yaw]); re[3] = wrap_quarter(re[3])
            L["obj_raw_err"].append(re)
            vt = np.array([*truth.cube_vel, 0.0])
            L["obj_vel_err"].append(info["obj_vel"] - vt); L["obj_vel_hat"].append(info["obj_vel"]); L["obj_vel_true"].append(vt)
        else:
            L["obj_err"].append(np.full(4, np.nan)); L["obj_sig"].append(np.full(4, np.nan)); L["obj_nees"].append(np.nan)
            L["obj_raw_err"].append(np.full(4, np.nan))
            L["obj_vel_err"].append(np.full(4, np.nan)); L["obj_vel_hat"].append(np.full(4, np.nan))
            L["obj_vel_true"].append(np.full(4, np.nan))
        L["nis_obj"].append(info["nis_obj"])
        if info["phase"] == Phase.GRASP and grasp_err is None:
            Rt = truth.tcp_rot
            grasp_err = Rt.T @ (truth.cube_pos - truth.tcp_pos)
            oe = info["obj"] - np.array([*truth.cube_pos, truth.cube_yaw]); oe[3] = wrap_quarter(oe[3])
            obj_err_at_grasp = oe
            re = info["obj_raw"] - np.array([*truth.cube_pos, truth.cube_yaw]); re[3] = wrap_quarter(re[3])
            raw_err_at_grasp = re
            vel_err_at_grasp = float(np.linalg.norm(info["obj_vel"][:3] - np.asarray(truth.cube_vel)))
        if tc.frames and k % 4 == 0:
            frames.append(dict(t=info["t"], ext=plant.render_view(free=dict(lookat=[0.21, -0.02, 0.15], distance=0.78,
                                                                           azimuth=132, elevation=-20)),
                               cam=plant.render_view("wrist_cam", 320, 240), phase=info["phase"]))
        # -----------------------------------------------------------------
        if info["phase"] in (Phase.DONE, Phase.FAILED):
            if done_k is None:
                done_k = k
            if k - done_k > 30:
                break
        obs = plant.step(u, grip)

    A = {k: np.array(v) for k, v in L.items()}
    cube_final = truth.cube_pos
    lifted = bool(cube_final[2] > 0.12 and truth.holding)
    success = bool(lifted and ctl.phase == Phase.DONE and not diverged)
    moving = A["phase"] < Phase.DONE
    track = A["q"] - A["q_ref"]
    res = dict(
        seed=tc.seed, success=success, lifted=lifted, diverged=diverged, final_phase=int(ctl.phase), time=float(A["t"][-1]),
        cube_xy=list(pc.cube_xy), cube_yaw=float(pc.cube_yaw),
        grasp_err=None if grasp_err is None else grasp_err.tolist(),
        grasp_err_norm=np.nan if grasp_err is None else float(np.linalg.norm(grasp_err)),
        obj_err_at_grasp=None if obj_err_at_grasp is None else obj_err_at_grasp.tolist(),
        obj_xy_err_at_grasp=np.nan if obj_err_at_grasp is None else float(np.linalg.norm(obj_err_at_grasp[:2])),
        raw_xy_err_at_grasp=np.nan if raw_err_at_grasp is None else float(np.linalg.norm(raw_err_at_grasp[:2])),
        cube_moved_before_grasp=float(np.linalg.norm((cube_final if grasp_err is None else cube0)[:2] - cube0[:2])),
        track_rmse=float(np.sqrt((track[moving] ** 2).mean())),
        tcp_track_rmse=float(np.sqrt(((A["tcp"] - A["p_ref"][:, :3])[moving] ** 2).sum(1).mean())),
        q_est_rmse=float(np.sqrt(((A["q_hat"] - A["q"]) ** 2).mean())),
        qd_est_rmse=float(np.sqrt(((A["qd_hat"] - A["qd"]) ** 2).mean())),
        nees_joint_mean=float(np.mean(A["nees_joint"][5:])), nis_joint_mean=float(np.nanmean(A["nis_joint"][5:])),
        obj_nees_mean=float(np.nanmean(A["obj_nees"])) if np.isfinite(A["obj_nees"]).any() else np.nan,
        nis_obj_mean=float(np.nanmean(A["nis_obj"])) if np.isfinite(A["nis_obj"]).any() else np.nan,
        effort_rms=float(np.sqrt((A["tau"][moving] ** 2).mean())),
        u_chatter=float(np.sqrt((np.diff(A["u"], axis=0)[moving[1:]] ** 2).mean())),
        obj_updates=int(ctl.obj.n_updates), obj_rejected=int(ctl.obj.n_rejected),
        obj_speed=float(np.linalg.norm(pc.object_motion[1:3])) if (pc.object_motion and pc.object_motion[0] == "linear") else 0.0,
        obj_vel_rmse=float(np.sqrt(np.nanmean(np.nansum(A["obj_vel_err"][:, :3] ** 2, axis=1)))) if np.isfinite(A["obj_vel_err"]).any() else np.nan,
        obj_vel_err_at_grasp=vel_err_at_grasp,
        grasp_check=None if plant.grasp_check is None else dict(rel=np.asarray(plant.grasp_check["rel"]).tolist(),
                                                              psi=float(plant.grasp_check["psi"]), ok=plant.grasp_check["ok"]),
    )
    if tc.keep_log:
        res["log"] = A
    if tc.frames:
        res["frames"] = frames
    plant.close()
    return res
