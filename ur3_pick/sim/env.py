"""Plant: MuJoCo simulation of UR3e + wrist RGB-D camera + gripper, exposing ONLY sensor data to the controller.

The controller side (core/) receives `Observation` objects. Ground truth is available only through
`Plant.truth`, which is imported by eval/ code and never by core/ (enforced by tests/test_firewall.py).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List
import numpy as np
import mujoco as mj

from . import scene as S

CTRL_DT = 0.01          # controller period [s] (100 Hz)
N_SUB = 5               # physics steps per control period (timestep 0.002)
CAM_PERIOD = 10         # camera every 10 controller steps (10 Hz)
V_REL_CAPTURE = 0.05    # max tool-object relative speed the jaws can capture [m/s]
NJ = 6


@dataclass
class PlantConfig:
    noise_scale: float = 1.0              # multiplies encoder, depth and pixel noise
    enc_sigma: float = 1e-3               # encoder noise std [rad] at noise_scale = 1
    depth_sigma0: float = 0.0015          # depth noise std at 0 range [m]
    depth_sigma2: float = 0.010           # depth noise std growth: sigma = s0 + s2 * z^2
    pix_sigma: float = 4.0                # RGB noise std [grey levels]
    depth_dropout_p: float = 0.002        # per-pixel invalid depth probability
    cube_xy: Tuple[float, float] = (0.22, -0.22)
    cube_yaw: float = 0.0
    camera_dropouts: Sequence[Tuple[float, float]] = ()    # (t0, t1) windows with no camera frame
    disturbance: Optional[Tuple[float, float, Tuple[float, float, float]]] = None  # (t0, dur, force N)
    payload_kg: float = 0.0               # unmodelled extra mass on wrist_3
    board: bool = False                   # calibration board scene (no cube)
    # Object motion. None = static. ("linear", vx, vy) slides at constant velocity;
    # ("sine", amp, freq, axis) oscillates so the object stays inside the workspace for longer.
    object_motion: Optional[tuple] = None
    seed: int = 0


@dataclass
class Observation:
    t: float
    q_meas: np.ndarray                    # (6,) noisy encoder reading
    rgb: Optional[np.ndarray] = None      # (H,W,3) uint8, present at camera ticks only
    depth: Optional[np.ndarray] = None    # (H,W) float32 metres (0 = invalid)


class GroundTruth:
    """Simulator ground truth. For evaluation only - never import from core/."""

    def __init__(self, plant: "Plant"):
        self._p = plant

    @property
    def q(self): return self._p.data.qpos[self._p.qadr].copy()
    @property
    def qd(self): return self._p.data.qvel[self._p.vadr].copy()
    @property
    def tcp_pos(self): return self._p.data.site_xpos[self._p.tcp_id].copy()
    @property
    def tcp_rot(self): return self._p.data.site_xmat[self._p.tcp_id].reshape(3, 3).copy()
    @property
    def cube_pos(self): return self._p.data.xpos[self._p.cube_id].copy()
    @property
    def cube_yaw(self):
        R = self._p.data.xmat[self._p.cube_id].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))
    @property
    def cube_vel(self):
        """True object velocity: the analytic commanded value for a driven object, else from the simulator."""
        if self._p.cfg.object_motion is not None and not self._p.holding:
            return self._p._obj_v_cmd.copy()
        return self._p.data.cvel[self._p.cube_id][3:].copy()
    @property
    def holding(self): return self._p.holding
    @property
    def u_effort(self): return self._p.data.actuator_force.copy()


_MODEL_CACHE: dict = {}       # one compiled MjModel + Renderer per scene variant, per process
_PRISTINE: dict = {}          # the model fields a Plant mutates, as compiled, soeach reset can restore them


def _cached_model(board: bool):
    """Compile each scene variant once per process.

    A mesh-heavy MjModel plus its offscreen renderer costs ~75 MB that the allocator does not hand back,
    so recompiling per trial made long Monte Carlo runs grow without bound. The initial cube pose baked into
    the XML is irrelevant: Plant.reset() writes the cube's qpos anyway.
    """
    key = bool(board)
    if key not in _MODEL_CACHE:
        m = S.load_model(board=board, with_cube=not board)
        r = mj.Renderer(m, S.IMG_H, S.IMG_W)
        _MODEL_CACHE[key] = (m, r)
        _PRISTINE[key] = dict(body_mass=m.body_mass.copy(), body_inertia=m.body_inertia.copy(),
                              geom_pos=m.geom_pos.copy(), geom_contype=m.geom_contype.copy(),
                              geom_conaffinity=m.geom_conaffinity.copy(),
                              eq_data=m.eq_data.copy() if m.neq else None)
    return _MODEL_CACHE[key]


class Plant:
    def __init__(self, cfg: PlantConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.model, self.renderer = _cached_model(cfg.board)
        self._pristine = _PRISTINE[bool(cfg.board)]
        self.data = mj.MjData(self.model)
        m = self.model
        names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.qadr = np.array([m.joint(n).qposadr[0] for n in names])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in names])
        self.tcp_id = m.site("tcp").id
        self.w3_id = m.body("wrist_3_link").id
        self.cam_id = m.camera("wrist_cam").id
        self.has_cube = not cfg.board
        self.cube_id = m.body("cube").id if self.has_cube else -1
        self.finger_ids = [m.geom("finger0").id, m.geom("finger1").id]
        self._finger_pos0 = [m.geom_pos[g].copy() for g in self.finger_ids]
        self._finger_contype = [(int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in self.finger_ids]
        self.truth = GroundTruth(self)
        self.holding = False
        self._grip_cmd = 0.0
        self.grasp_check = None
        self._obj_v_cmd = np.zeros(3)
        self._aperture = 0.04
        self.k = 0
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self):
        m, d = self.model, self.data
        # the MjModel is shared across trials in this process, so undo the previous trial's model edits
        pr = self._pristine
        m.body_mass[:] = pr["body_mass"]
        m.body_inertia[:] = pr["body_inertia"]
        m.geom_pos[:] = pr["geom_pos"]
        m.geom_contype[:] = pr["geom_contype"]
        m.geom_conaffinity[:] = pr["geom_conaffinity"]
        if pr["eq_data"] is not None:
            m.eq_data[:] = pr["eq_data"]
        if self.cfg.payload_kg:
            m0 = m.body_mass[self.w3_id]
            m.body_mass[self.w3_id] = m0 + self.cfg.payload_kg
            m.body_inertia[self.w3_id] *= (m0 + self.cfg.payload_kg) / m0
        mj.mj_resetData(m, d)
        d.qpos[self.qadr] = S.HOME_Q
        d.ctrl[:NJ] = S.HOME_Q
        if self.has_cube:
            a = m.jnt_qposadr[m.body_jntadr[self.cube_id]]
            d.qpos[a:a + 3] = [self.cfg.cube_xy[0], self.cfg.cube_xy[1], S.TABLE_Z + S.CUBE_HALF]
            d.qpos[a + 3:a + 7] = [np.cos(self.cfg.cube_yaw / 2), 0, 0, np.sin(self.cfg.cube_yaw / 2)]
            m.eq_active0[0] = 0
            d.eq_active[0] = 0
        self.holding = False
        self._set_aperture(0.04)
        for g, (ct, ca) in zip(self.finger_ids, self._finger_contype):
            m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        mj.mj_forward(m, d)
        for _ in range(int(0.5 / (CTRL_DT / N_SUB))):   # let the servos settle at home
            mj.mj_step(m, d)
        d.time = 0.0
        self.k = 0
        self._last_obs = self._observe()
        return self._last_obs

    def close(self):
        """Release this trial's MjData. The model and renderer are process-wide and stay cached."""
        self.data = None

    # ------------------------------------------------------------------ gripper
    def _set_aperture(self, a: float):
        self._aperture = a
        for i, sgn in enumerate((+1, -1)):
            p = self._finger_pos0[i].copy()
            # finger centre moves along tool x == wrist_3 x axis (R_W3_TOOL[:,0] = x)
            p[0] = sgn * a
            self.model.geom_pos[self.finger_ids[i]] = p

    def _object_state(self, t: float):
        """Analytic (position offset, velocity) of the driven object at time t. None = free/static.

        Modelling assumption: a driven object is moved KINEMATICALLY, as if fixed to a conveyor belt, so its
        trajectory is exact and its true velocity is known analytically (which is what the velocity estimate
        is scored against). The consequence is that the gripper cannot nudge the object before it is grasped.
        """
        m = self.cfg.object_motion
        if m is None:
            return None
        if m[0] == "linear":
            v = np.array([m[1], m[2], 0.0])
            return v * t, v
        if m[0] == "sine":
            _, amp, freq, axis = m
            w = 2 * np.pi * freq
            d = np.zeros(3); v = np.zeros(3)
            d[axis] = amp * np.sin(w * t)
            v[axis] = amp * w * np.cos(w * t)
            return d, v
        raise ValueError(m[0])

    def _drive_object(self):
        """Hold the driven object exactly on its commanded trajectory while it is not grasped."""
        st = self._object_state(self.data.time)
        if st is None or self.holding or not self.has_cube:
            return
        d_xy, v = st
        m, d = self.model, self.data
        qadr = m.jnt_qposadr[m.body_jntadr[self.cube_id]]
        vadr = m.jnt_dofadr[m.body_jntadr[self.cube_id]]
        d.qpos[qadr:qadr + 2] = np.array(self.cfg.cube_xy) + d_xy[:2]
        d.qvel[vadr:vadr + 3] = v
        d.qvel[vadr + 3:vadr + 6] = 0.0
        self._obj_v_cmd = v

    def _try_grasp(self):
        """Jaw-capture model: close fingers on the cube if it lies inside the jaw envelope."""
        if not self.has_cube or self.holding:
            return
        m, d = self.model, self.data
        Rt = d.site_xmat[self.tcp_id].reshape(3, 3)
        rel = Rt.T @ (d.xpos[self.cube_id] - d.site_xpos[self.tcp_id])
        Rc = d.xmat[self.cube_id].reshape(3, 3)
        Rrel = Rt.T @ Rc
        psi = np.arctan2(Rrel[1, 0], Rrel[0, 0])
        psi = (psi + np.pi / 4) % (np.pi / 2) - np.pi / 4
        half_width = S.CUBE_HALF * (abs(np.cos(psi)) + abs(np.sin(psi)))
        # relative velocity matters: jaws closing on an object that is sliding past knock it away instead of
        # capturing it. This is what makes the tracker's velocity estimate necessary rather than decorative.
        v_obj = self._obj_v_cmd if self.cfg.object_motion is not None else self.data.cvel[self.cube_id][3:]
        v_rel = v_obj - self.data.cvel[self.w3_id][3:]
        speed_rel = float(np.linalg.norm(v_rel[:2]))
        ok = (abs(rel[0]) < 0.035 - half_width and abs(rel[1]) < 0.012 and abs(rel[2]) < 0.015
              and abs(Rrel[2, 2]) > 0.95 and speed_rel < V_REL_CAPTURE)
        self.grasp_check = dict(rel=rel, psi=psi, speed_rel=speed_rel, ok=bool(ok))
        if not ok:
            return
        # engage weld with the current relative pose (no jump)
        Rb = d.xmat[self.w3_id].reshape(3, 3)
        pb, pc = d.xpos[self.w3_id], d.xpos[self.cube_id]
        rel_pos = Rb.T @ (pc - pb)
        qb, qc, qi = d.xquat[self.w3_id], d.xquat[self.cube_id], np.zeros(4)
        mj.mju_negQuat(qi, qb)
        rel_q = np.zeros(4)
        mj.mju_mulQuat(rel_q, qi, qc)
        m.eq_data[0, :3] = 0.0
        m.eq_data[0, 3:6] = rel_pos
        m.eq_data[0, 6:10] = rel_q
        m.eq_data[0, 10] = 1.0
        d.eq_active[0] = 1
        self.holding = True
        for g in self.finger_ids:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0

    # ------------------------------------------------------------------ stepping
    def step(self, u: np.ndarray, grip: float = 0.0) -> Observation:
        """Apply joint position targets u (6,) for one controller period; grip in {0 open, 1 closed}."""
        m, d = self.model, self.data
        d.ctrl[:NJ] = u
        dist = self.cfg.disturbance
        for _ in range(N_SUB):
            t = d.time
            d.xfrc_applied[self.w3_id, :3] = 0.0
            if dist is not None and dist[0] <= t < dist[0] + dist[1]:
                d.xfrc_applied[self.w3_id, :3] = dist[2]
            self._drive_object()
            mj.mj_step(m, d)
        # gripper kinematics (jaws close at 0.3 m/s)
        target = 0.025 if grip > 0.5 else 0.04
        a = self._aperture + np.clip(target - self._aperture, -0.003, 0.003) if not self.holding else 0.025
        if grip > 0.5 and not self.holding and abs(a - 0.025) < 1e-6:
            self._try_grasp()
        if grip < 0.5 and self.holding:  # release
            d.eq_active[0] = 0
            self.holding = False
            for g, (ct, ca) in zip(self.finger_ids, self._finger_contype):
                m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        self._set_aperture(a)
        self.k += 1
        self._last_obs = self._observe()
        return self._last_obs

    # ------------------------------------------------------------------ sensors
    def _observe(self) -> Observation:
        s = self.cfg.noise_scale
        q = self.data.qpos[self.qadr] + self.rng.normal(0, self.cfg.enc_sigma * s, NJ)
        obs = Observation(t=float(self.data.time), q_meas=q)
        if self.k % CAM_PERIOD == 0:
            t = self.data.time
            if not any(a <= t < b for a, b in self.cfg.camera_dropouts):
                obs.rgb, obs.depth = self._render()
        return obs

    def _render(self):
        s = self.cfg.noise_scale
        r = self.renderer
        r.update_scene(self.data, camera="wrist_cam")
        rgb = r.render().astype(np.float32)
        r.enable_depth_rendering()
        r.update_scene(self.data, camera="wrist_cam")
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        rgb += self.rng.normal(0, self.cfg.pix_sigma * s, rgb.shape)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        sig = (self.cfg.depth_sigma0 + self.cfg.depth_sigma2 * depth ** 2) * s
        depth = depth + self.rng.normal(0, 1, depth.shape).astype(np.float32) * sig
        depth[self.rng.random(depth.shape) < self.cfg.depth_dropout_p * s] = 0.0
        return rgb, depth.astype(np.float32)

    def render_view(self, camera: str = "wrist_cam", width: int = 640, height: int = 480, free: Optional[dict] = None):
        """Visual (noise-free) render for videos. `free` = dict(lookat, distance, azimuth, elevation)."""
        key = ("view", width, height)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = mj.Renderer(self.model, height, width)
        r = _MODEL_CACHE[key]
        if True:
            if free:
                cam = mj.MjvCamera()
                cam.lookat[:] = free["lookat"]; cam.distance = free["distance"]
                cam.azimuth = free["azimuth"]; cam.elevation = free["elevation"]
                r.update_scene(self.data, camera=cam)
            else:
                r.update_scene(self.data, camera=camera)
            return r.render().copy()
