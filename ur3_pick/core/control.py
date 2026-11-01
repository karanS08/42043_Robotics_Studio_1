"""Control: PBVS pick state machine + Cartesian reference filter + DLS IK + LQR joint tracker on ESTIMATED state.

Everything here consumes `Observation` (encoders + RGB-D) only. No simulator state is imported or accessed.

Control law (per 10 ms):
    x_hat = [q_hat, qd_hat]            from JointEKF
    x_ref = [q_ref, qd_ref]            from Cartesian reference filter -> IK
    u     = u_ff(q_ref, qd_ref, qdd_ref) - K (x_hat - x_ref)
u is the vector of joint position targets of the UR3e's inner servo loop (tau = kp (u - q) - kd qd).
u_ff inverts that servo + rigid-body model:  u_ff = q_ref + (M qdd_ref + c(q_ref, qd_ref) + g(q_ref) + kd qd_ref) / kp.
K is the discrete LQR gain for the 100 Hz linearisation (Ad, Bd) of the same model about the current estimate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np
import mujoco as mj
from scipy.linalg import solve_discrete_are
from scipy.spatial.transform import Rotation as Rot

from .kinematics import RobotModel
from .estimation import JointEKF, ObjectKF, NoiseModel
from .sensing import Calibration, CubeDetector, wrap_quarter

NJ = 6
CTRL_DT = 0.01
HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
QD_REF_MAX = 2.0        # rad/s ceiling on the joint reference rate


class Phase(IntEnum):
    SCAN = 0
    PREGRASP = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    DONE = 5
    FAILED = 6


@dataclass
class ControllerConfig:
    # LQR weights (Bryson's rule: 1 / max_acceptable^2)
    q_max: float = 0.005         # rad   -> Q_q = 1 / q_max^2
    qd_max: float = 1.0          # rad/s -> Q_v = 1 / qd_max^2
    u_max: float = 0.02          # rad   -> R   = 1 / u_max^2
    use_lqr: bool = True         # False = inner servo + feed-forward only (baseline)
    # Inner-loop model. "mujoco" inverts the simulated position servo (tau = kp(u-q) - kd qd) and is only
    # valid against that plant. A real UR3e has its own servoJ loop, so on hardware either identify its
    # gains (scripts/hw_identify_servo.py writes servo_kp/servo_kd) or use "none", which sends the joint
    # reference straight through and keeps only the LQR correction on the estimate.
    feedforward: str = "mujoco"  # "mujoco" | "gains" | "none"
    servo_kp: Optional[tuple] = None    # used when feedforward == "gains"
    servo_kd: Optional[tuple] = None
    lqr_period: int = 5          # recompute gain every N control steps
    # estimator tuning
    enc_sigma: float = 1e-3
    accel_sigma: float = 2.0
    enc_r_scale: float = 1.0     # multiplies the encoder measurement covariance R assumed by the EKF (tuning study)
    obj_r_scale: float = 1.0     # multiplies the camera measurement covariance R assumed by the object KF
    assumed_noise_scale: float = 1.0
    use_object_kf: bool = True   # False = use the latest raw detection (baseline)
    obj_model: str = "cv"        # "cv" = constant-velocity tracker, "static" = original static-object filter
    obj_accel_sigma: float = 0.05   # m/s^2 process noise of the object tracker (tuning study)
    z_estimator: str = "median"  # cube-height estimator in the detector ("p90" = first, biased version)
    # reference filter
    omega: float = 5.0           # rad/s, critically damped
    v_max: float = 0.25          # m/s
    # task geometry
    scan_poses: tuple = ((0.22, 0.10, 0.30), (0.22, -0.08, 0.30))   # first covers the whole workspace; second = search fallback
    search_dwell: float = 1.0    # s to wait at a scan pose without detection before moving to the next
    pregrasp_height: float = 0.10
    # moving-object interception
    track_object_velocity: bool = True   # False = ignore the estimated velocity (ablation)
    lead_time: float = 0.25              # s of look-ahead while approaching a moving object (0 once descending)
    v_match_tol: float = 0.005           # m/s: estimated tool-object speed mismatch allowed before grasping
    lift_height: float = 0.20
    timeout: float = 16.0
    # reachable Cartesian box the goal is clamped into. The arm must never chase an object out of its
    # workspace: the IK then has no solution, the joint reference jumps and the loop can diverge.
    ws_x: tuple = (0.13, 0.31)
    ws_y: tuple = (-0.26, 0.26)
    ws_z: tuple = (0.0, 0.36)
    lost_timeout: float = 2.5            # s without a usable detection before a MOVING track is called stale
    stale_speed: float = 0.015           # m/s: below this the object counts as static and coasting is safe
    payload_model_kg: float = 0.0


class ReferenceFilter:
    """Critically damped 2nd-order filter on [x, y, z, yaw] with a speed limit: smooth, bounded references even when the goal jumps."""

    def __init__(self, p0: np.ndarray, omega: float, v_max: float):
        self.p = p0.astype(float).copy()
        self.v = np.zeros(4)
        self.omega, self.v_max = omega, v_max

    def step(self, goal: np.ndarray, goal_vel: Optional[np.ndarray] = None, dt: float = CTRL_DT):
        """Second-order tracker with velocity feed-forward.

        a = w^2 (goal - p) + 2w (goal_vel - v). Without the feed-forward term a critically damped filter
        tracking a ramp settles with a lag of 2*v/w - 16 mm at 0.04 m/s and w = 5, far more than the grasp
        tolerance. With it the steady-state lag on a constant-velocity goal is zero.
        """
        gv = np.zeros(4) if goal_vel is None else goal_vel
        e = goal - self.p
        a = self.omega ** 2 * e + 2 * self.omega * (gv - self.v)
        self.v = self.v + a * dt
        sp = np.linalg.norm(self.v[:3])
        if sp > self.v_max:
            self.v[:3] *= self.v_max / sp
        self.p = self.p + self.v * dt
        return self.p.copy(), self.v.copy()


class LQRTracker:
    def __init__(self, robot: RobotModel, cfg: ControllerConfig):
        self.robot, self.cfg = robot, cfg
        self.Qx = np.diag([1 / cfg.q_max ** 2] * NJ + [1 / cfg.qd_max ** 2] * NJ)
        self.Ru = np.eye(NJ) / cfg.u_max ** 2
        self.K = np.zeros((NJ, 2 * NJ))
        self._d = mj.MjData(robot.m)
        self.k = 0

    def update_gain(self, q_hat: np.ndarray):
        """Gain-scheduled discrete LQR on the analytic servo + rigid-body linearisation about the current estimate."""
        if not self.cfg.use_lqr:
            return
        if self.k % self.cfg.lqr_period == 0:
            try:
                Ad, Bd = self.robot.discrete_model(q_hat, CTRL_DT)
                P = solve_discrete_are(Ad, Bd, self.Qx, self.Ru)
                self.K = np.linalg.solve(self.Ru + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
            except Exception:
                pass   # keep previous gain
        self.k += 1

    def feedforward(self, q, qd, qdd) -> np.ndarray:
        """Position target that an inner position servo needs in order to realise (q, qd, qdd).

        mode "none": the reference itself. Correct whenever the inner loop is a black box that already does
        its own gravity and dynamics compensation, which is the case for the UR controller behind servoJ.
        """
        mode = self.cfg.feedforward
        if mode == "none":
            return np.asarray(q, float)
        r, d = self.robot, self._d
        d.qpos[r.qadr], d.qvel[r.vadr], d.qacc[r.vadr] = q, qd, qdd
        mj.mj_inverse(r.m, d)
        tau = d.qfrc_inverse[r.vadr]
        kp = np.asarray(self.cfg.servo_kp, float) if mode == "gains" else r.kp
        kd = np.asarray(self.cfg.servo_kd, float) if mode == "gains" else r.kd
        return q + (tau + kd * qd) / kp

    def control(self, x_hat, q_ref, qd_ref, qdd_ref) -> np.ndarray:
        u = self.feedforward(q_ref, qd_ref, qdd_ref)
        if self.cfg.use_lqr:
            u = u - self.K @ (x_hat - np.concatenate([q_ref, qd_ref]))
        return u


class PickController:
    """Full pipeline: estimation -> PBVS goal -> reference -> IK -> LQR. Call step(obs) every 10 ms."""

    def __init__(self, cfg: Optional[ControllerConfig] = None, calib: Optional[Calibration] = None,
                 noise: Optional[NoiseModel] = None):
        self.cfg = cfg or ControllerConfig()
        self.robot = RobotModel(payload_kg=self.cfg.payload_model_kg)
        self.calib = calib or Calibration.load()
        self.detector = CubeDetector(self.robot, self.calib, z_estimator=self.cfg.z_estimator)
        self.ekf = JointEKF(self.robot, HOME_Q,
                            enc_sigma=self.cfg.enc_sigma * self.cfg.assumed_noise_scale * np.sqrt(self.cfg.enc_r_scale),
                            accel_sigma=self.cfg.accel_sigma)
        self.obj = ObjectKF(noise or NoiseModel.load(), noise_scale=self.cfg.assumed_noise_scale,
                            r_scale=self.cfg.obj_r_scale, model=self.cfg.obj_model,
                            accel_sigma=self.cfg.obj_accel_sigma)
        self.lqr = LQRTracker(self.robot, self.cfg)
        p0, self.R_home = self.robot.fk(HOME_Q)
        self.ref = ReferenceFilter(np.array([*p0, 0.0]), self.cfg.omega, self.cfg.v_max)
        self.q_ref = HOME_Q.copy()
        self.qd_ref = np.zeros(NJ)
        self.u_prev: Optional[np.ndarray] = None
        self.phase = Phase.SCAN
        self.phase_t0 = 0.0
        self.updates_at_phase = 0
        self.grip = 0.0
        self.first = True
        self.raw_obj: Optional[np.ndarray] = None
        self.scan_idx = 0
        self.scan_arrived: Optional[float] = None
        self.goal = np.array([*self.cfg.scan_poses[0], 0.0])
        self.grasp_vel = np.zeros(4)
        self.goal_clamped = False
        self.last_meas_t: Optional[float] = None

    # ------------------------------------------------------------------ helpers
    def _obj_state(self) -> Optional[np.ndarray]:
        if self.cfg.use_object_kf:
            return self.obj.pose
        return self.raw_obj

    def _set_phase(self, ph: Phase, t: float):
        self.phase, self.phase_t0 = ph, t
        self.updates_at_phase = self.obj.n_updates

    def _R_des(self, yaw: float) -> np.ndarray:
        return Rot.from_euler("z", yaw).as_matrix() @ self.R_home

    # ------------------------------------------------------------------ main step
    def step(self, obs):
        cfg, t = self.cfg, obs.t
        # ---------------- 1. estimation (encoders -> EKF, RGB-D -> object KF)
        if self.first:
            self.ekf.x[:NJ] = obs.q_meas
            self.first = False
        else:
            self.ekf.predict(self.u_prev)
            self.ekf.update(obs.q_meas)
        new_meas = False
        if self.phase <= Phase.DESCEND:
            self.obj.predict()          # every control step: 100 Hz prediction against a 10 Hz camera
        if obs.rgb is not None and self.phase <= Phase.DESCEND:
            m = self.detector.detect(obs.rgb, obs.depth, self.ekf.q)
            if m is not None:
                self.raw_obj = np.array([*m.pos, m.yaw])
                new_meas = self.obj.update(m)
        x_hat = self.ekf.x.copy()
        tcp_hat, _ = self.robot.fk(x_hat[:NJ])

        # ---------------- 2. PBVS task logic on estimates
        o = self._obj_state()
        ph = self.phase
        n_new = self.obj.n_updates - self.updates_at_phase
        # Goal of the current phase. For a moving object the goal is where the estimator says the object WILL
        # be (pose + velocity * lead), and the goal velocity is fed forward so the tool matches its speed.
        o_vel = self.obj.vel if (cfg.track_object_velocity and self.cfg.use_object_kf) else np.zeros(4)
        if o is not None:
            # Lead only while flying to the object: aim at where it WILL be, so the approach intercepts it.
            # Once descending, the tool must sit ON the object (velocity feed-forward already removes the
            # tracking lag); leading here would place the jaws lead_time * speed ahead of the cube.
            lead = cfg.lead_time if ph == Phase.PREGRASP else 0.0
            o_lead = o + o_vel * lead
            o_lead[3] = wrap_quarter(o_lead[3])
            yaw_goal = self.ref.p[3] + wrap_quarter(o_lead[3] - self.ref.p[3])
        goal_vel = np.zeros(4)
        if ph == Phase.SCAN:
            self.goal = np.array([*cfg.scan_poses[self.scan_idx], 0.0])
        elif ph == Phase.PREGRASP:
            self.goal = np.array([o_lead[0], o_lead[1], o_lead[2] + cfg.pregrasp_height, yaw_goal])
            goal_vel = o_vel.copy()
        elif ph == Phase.DESCEND:
            self.goal = np.array([o_lead[0], o_lead[1], o_lead[2], yaw_goal])
            goal_vel = o_vel.copy()
        elif ph == Phase.GRASP:
            self.grip = 1.0
            # keep moving with the object while the jaws close, otherwise it slides out of them
            self.grasp_target[:3] += self.grasp_vel[:3] * CTRL_DT
            self.goal = self.grasp_target.copy()
            goal_vel = self.grasp_vel.copy()
        elif ph == Phase.LIFT:
            self.goal = self.grasp_target.copy()
            self.goal[2] += cfg.lift_height
        # clamp into the reachable workspace, and record whether the object itself has left it
        goal_raw = self.goal.copy()
        self.goal[0] = float(np.clip(self.goal[0], *cfg.ws_x))
        self.goal[1] = float(np.clip(self.goal[1], *cfg.ws_y))
        self.goal[2] = float(np.clip(self.goal[2], *cfg.ws_z))
        self.goal_clamped = bool(np.any(np.abs(goal_raw[:3] - self.goal[:3]) > 1e-9))
        if self.goal_clamped:
            goal_vel = np.zeros(4)       # do not keep accelerating toward an unreachable point
        tcp_err = np.linalg.norm(tcp_hat - self.goal[:3])
        ref_speed = np.linalg.norm(self.ref.v[:3] - goal_vel[:3])   # speed relative to the goal
        yaw_err = abs(self.ref.p[3] - self.goal[3])
        # transitions (all conditions are evaluated on estimated quantities)
        if ph == Phase.SCAN:
            if o is not None and tcp_err < 0.006 and self.obj.n_updates >= 5:
                self._set_phase(Phase.PREGRASP, t)
            elif tcp_err < 0.006 and o is None:               # nothing seen from here: search the next scan pose
                self.scan_arrived = t if self.scan_arrived is None else self.scan_arrived
                if t - self.scan_arrived > cfg.search_dwell:
                    self.scan_idx = (self.scan_idx + 1) % len(cfg.scan_poses)
                    self.scan_arrived = None
        elif ph == Phase.PREGRASP:
            if tcp_err < 0.004 and ref_speed < 0.01 and yaw_err < 0.02 and (n_new >= 5 or t - self.phase_t0 > 4.0):
                self._set_phase(Phase.DESCEND, t)
        elif ph == Phase.DESCEND:
            # speed mismatch between the tool and the object, both estimated
            v_mismatch = float(np.linalg.norm(self.ref.v[:3] - o_vel[:3]))
            if tcp_err < 0.003 and v_mismatch < cfg.v_match_tol and yaw_err < 0.02:
                self.grasp_target = self.goal.copy()
                self.grasp_vel = o_vel.copy()
                self._set_phase(Phase.GRASP, t)
        elif ph == Phase.GRASP:
            if t - self.phase_t0 > 0.4:
                self._set_phase(Phase.LIFT, t)
        elif ph == Phase.LIFT:
            if tcp_err < 0.005 and ref_speed < 0.01 and t - self.phase_t0 > 0.5:
                self._set_phase(Phase.DONE, t)
        if new_meas:
            self.last_meas_t = t
        if (self.phase in (Phase.PREGRASP, Phase.DESCEND)
                and self.goal_clamped and o is not None
                and not (cfg.ws_x[0] <= o[0] <= cfg.ws_x[1] and cfg.ws_y[0] <= o[1] <= cfg.ws_y[1])):
            self._set_phase(Phase.FAILED, t)        # object has left the reachable workspace
        # Extrapolating a MOVING object with no corrections goes stale; a static one does not, so the guard
        # is on the estimated speed, not merely on it being non-zero (it never is exactly).
        if (self.phase in (Phase.PREGRASP, Phase.DESCEND) and self.last_meas_t is not None
                and t - self.last_meas_t > cfg.lost_timeout
                and float(np.linalg.norm(self.obj.vel[:3])) > cfg.stale_speed):
            self._set_phase(Phase.FAILED, t)
        if t > cfg.timeout and self.phase not in (Phase.DONE, Phase.FAILED):
            self._set_phase(Phase.FAILED, t)

        # ---------------- 3. reference generation: Cartesian filter -> IK -> joint reference
        p_ref, v_ref = self.ref.step(self.goal, goal_vel)
        q_new, ik_ok = self.robot.ik(self.q_ref, p_ref[:3], self._R_des(p_ref[3]), iters=8)
        if not np.all(np.isfinite(q_new)):
            q_new = self.q_ref.copy()
        # rate limit: the joint reference may never jump, whatever the IK or the estimator returns
        q_new = np.clip(q_new, self.q_ref - QD_REF_MAX * CTRL_DT, self.q_ref + QD_REF_MAX * CTRL_DT)
        qd_new = (q_new - self.q_ref) / CTRL_DT
        qdd_ref = (qd_new - self.qd_ref) / CTRL_DT
        self.q_ref, self.qd_ref = q_new, qd_new

        # ---------------- 4. LQR on the estimated state
        self.lqr.update_gain(x_hat[:NJ])
        u = self.lqr.control(x_hat, self.q_ref, self.qd_ref, qdd_ref)
        self.u_prev = u
        info = dict(t=t, phase=int(self.phase), x_hat=x_hat, P_joint=self.ekf.P.copy(),
                    q_ref=self.q_ref.copy(), qd_ref=self.qd_ref.copy(), p_ref=p_ref, goal=self.goal.copy(),
                    tcp_hat=tcp_hat, u=u.copy(), nis_joint=self.ekf.last_nis,
                    obj=None if self.obj.x is None else self.obj.pose.copy(),
                    obj_vel=self.obj.vel.copy(),
                    obj_P=None if self.obj.P is None else self.obj.P_total.copy(),
                    obj_raw=None if self.raw_obj is None else self.raw_obj.copy(),
                    new_meas=new_meas, nis_obj=self.obj.last_nis if new_meas else np.nan, ik_ok=ik_ok)
        return u, self.grip, info
