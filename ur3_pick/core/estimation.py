"""State estimation.

JointEKF  : 12-state [q, qdot] EKF. Prediction = the robot's own MuJoCo model (servo + rigid-body dynamics) stepped with the
            commanded position targets u; covariance propagated with the finite-difference Jacobian of that step
            (mjd_transitionFD). Measurement = noisy encoders q (velocity is NOT measured -> it is an observer problem).
ObjectKF  : 4-state [x y z yaw] linear KF for the static cube, fed by CubeDetector measurements with a range-dependent R.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import mujoco as mj

from .kinematics import RobotModel
from .sensing import CubeMeasurement, wrap_quarter, CFG_DIR

NJ = 6
N_SUB = 5
CTRL_DT = 0.01


class JointEKF:
    def __init__(self, robot: RobotModel, q0: np.ndarray, enc_sigma: float = 1e-3, accel_sigma: float = 4.0,
                 p0_q: float = 1e-3, p0_v: float = 0.05):
        self.robot = robot
        self.m = robot.m
        self.d = mj.MjData(self.m)           # private integrator of the model
        self.x = np.concatenate([q0, np.zeros(NJ)])
        self.P = np.diag([p0_q ** 2] * NJ + [p0_v ** 2] * NJ)
        self.R = np.eye(NJ) * enc_sigma ** 2
        self.accel_sigma = accel_sigma
        self.nx = 2 * NJ
        self.A_fd = np.zeros((self.nx, self.nx))
        self.B_fd = np.zeros((self.nx, NJ))
        self.last_nis = np.nan
        self.last_innov = np.zeros(NJ)

    @property
    def q(self): return self.x[:NJ]
    @property
    def qd(self): return self.x[NJ:]

    def _Q(self):
        dt, s2 = CTRL_DT, self.accel_sigma ** 2
        Q = np.zeros((self.nx, self.nx))
        for i in range(NJ):
            Q[i, i] = s2 * dt ** 4 / 4
            Q[i, NJ + i] = Q[NJ + i, i] = s2 * dt ** 3 / 2
            Q[NJ + i, NJ + i] = s2 * dt ** 2
        return Q

    def predict(self, u: np.ndarray):
        d, m = self.d, self.m
        va, qa = self.robot.vadr, self.robot.qadr
        d.qpos[qa] = self.x[:NJ]
        d.qvel[va] = self.x[NJ:]
        d.ctrl[:NJ] = u
        d.qacc_warmstart[:] = 0
        mj.mj_forward(m, d)
        mj.mjd_transitionFD(m, d, 1e-6, True, self.A_fd, self.B_fd, None, None)
        A1, B1 = self.A_fd, self.B_fd
        # 5 physics substeps per control period with zero-order-hold control: Ad = A1^5, Bd = sum A1^i B1
        Ad = np.linalg.matrix_power(A1, N_SUB)
        S_ = np.zeros_like(A1)
        Ai = np.eye(self.nx)
        for _ in range(N_SUB):
            S_ += Ai
            Ai = Ai @ A1
        Bd = S_ @ B1
        for _ in range(N_SUB):
            mj.mj_step(m, d)
        self.x = np.concatenate([d.qpos[qa], d.qvel[va]])
        self.P = Ad @ self.P @ Ad.T + self._Q()
        self.P = 0.5 * (self.P + self.P.T)
        self.Ad, self.Bd = Ad, Bd

    def update(self, q_meas: np.ndarray):
        H = np.hstack([np.eye(NJ), np.zeros((NJ, NJ))])
        y = q_meas - self.x[:NJ]
        S_ = H @ self.P @ H.T + self.R
        S_ = 0.5 * (S_ + S_.T)
        try:
            K = self.P @ H.T @ np.linalg.inv(S_)
        except np.linalg.LinAlgError:
            return                      # covariance has gone non-finite: skip rather than crash the run
        self.x = self.x + K @ y
        I = np.eye(self.nx)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T       # Joseph form
        self.last_innov = y
        self.last_nis = float(y @ np.linalg.solve(S_, y))


@dataclass
class NoiseModel:
    """Range-dependent measurement variance of the cube detector (fitted in scripts/characterise_noise.py)."""
    xy_a: float = 1e-6       # sigma_xy^2 = xy_a + xy_b * range^2   [m^2]
    xy_b: float = 1e-5
    z_a: float = 1e-6
    z_b: float = 1e-5
    yaw_a: float = 1e-3      # rad^2
    yaw_b: float = 1e-2
    bias_xy: float = 0.0     # variance of the slowly varying (pose-dependent) error that averaging cannot remove [m^2]
    bias_z: float = 0.0
    bias_yaw: float = 0.0

    @property
    def bias_cov(self) -> np.ndarray:
        return np.diag([self.bias_xy, self.bias_xy, self.bias_z, self.bias_yaw])

    @staticmethod
    def load(path: Optional[str] = None) -> "NoiseModel":
        path = path or os.path.join(CFG_DIR, "noise_model.json")
        if os.path.exists(path):
            d = json.load(open(path))
            # The file also carries provenance (source, n_poses, ...) that is not a model parameter;
            # keep only the fitted fields so a hardware-written file loads as readily as a simulated one.
            return NoiseModel(**{k: v for k, v in d.items() if k in NoiseModel.__dataclass_fields__})
        return NoiseModel()

    def R(self, rng: float, scale: float = 1.0) -> np.ndarray:
        r2 = rng ** 2
        sxy = self.xy_a + self.xy_b * r2
        sz = self.z_a + self.z_b * r2
        sy = self.yaw_a + self.yaw_b * r2
        return np.diag([sxy, sxy, sz, sy]) * scale ** 2


class ObjectKF:
    """Object tracker, state [x, y, z, yaw] (+ their rates in the constant-velocity model).

    model="cv"     : constant-velocity, 8 states. Prediction runs every control step (100 Hz) while the camera
                     only updates at 10 Hz, so the predicted velocity is what carries the estimate between
                     frames and through dropouts. This is the model used for a moving object.
    model="static" : the original 4-state static model, kept so the ablation can show what a moving object
                     does to a filter whose process model says the object cannot move.

    Only position and yaw are measured; the rates are never measured and must be reconstructed - the same
    observability argument as the joint EKF's velocity states.
    """

    GATE = 18.47          # chi2(4 dof), 99.9 %
    REINIT_AFTER = 8

    def __init__(self, noise: NoiseModel, q_pos: float = 1e-8, q_yaw: float = 1e-8, noise_scale: float = 1.0,
                 r_scale: float = 1.0, model: str = "cv", accel_sigma: float = 0.05, alpha_sigma: float = 2.0,
                 dt: float = CTRL_DT):
        self.noise, self.noise_scale, self.r_scale = noise, noise_scale, r_scale
        self.model = model
        self.nx = 8 if model == "cv" else 4
        self.dt = dt
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.n_updates = 0
        self.n_rejected = 0
        self.consec_rejected = 0
        self.n_reinit = 0
        self.last_nis = np.nan
        self.last_z: Optional[np.ndarray] = None    # last raw measurement (the "no filter" baseline)
        self.H = np.hstack([np.eye(4), np.zeros((4, 4))]) if model == "cv" else np.eye(4)
        if model == "cv":
            self.F = np.block([[np.eye(4), dt * np.eye(4)], [np.zeros((4, 4)), np.eye(4)]])
            # piecewise white acceleration: sigma_a on x,y,z and alpha on yaw
            s2 = np.diag([accel_sigma ** 2] * 3 + [alpha_sigma ** 2])
            self.Q = np.block([[s2 * dt ** 4 / 4, s2 * dt ** 3 / 2], [s2 * dt ** 3 / 2, s2 * dt ** 2]])
        else:
            self.F = np.eye(4)
            self.Q = np.diag([q_pos, q_pos, q_pos, q_yaw])

    # ------------------------------------------------------------------ accessors
    @property
    def initialised(self): return self.x is not None

    @property
    def pose(self) -> Optional[np.ndarray]:
        """[x, y, z, yaw] whichever model is in use."""
        return None if self.x is None else self.x[:4]

    @property
    def vel(self) -> np.ndarray:
        """[vx, vy, vz, yaw_rate]; identically zero for the static model."""
        if self.x is None or self.model != "cv":
            return np.zeros(4)
        return self.x[4:]

    def predict_ahead(self, dt: float) -> Optional[np.ndarray]:
        """Pose extrapolated dt seconds into the future - what an interception goal must aim at."""
        if self.x is None:
            return None
        p = self.x[:4] + self.vel * dt
        p[3] = wrap_quarter(p[3])
        return p

    @property
    def P_total(self) -> np.ndarray:
        """Reported covariance of the POSE block, including the calibration/view bias.

        A common-mode bias is not averaged away by a Kalman filter, so the white-noise covariance alone is
        over-confident; the fitted bias covariance is added to what the filter reports.
        """
        Ppose = self.P[:4, :4]
        return Ppose + self.noise.bias_cov * self.noise_scale ** 2

    # ------------------------------------------------------------------ filter
    def _init_from(self, z: np.ndarray, R: np.ndarray):
        if self.model == "cv":
            self.x = np.concatenate([z, np.zeros(4)])
            self.P = np.block([[R, np.zeros((4, 4))],
                               [np.zeros((4, 4)), np.diag([0.25, 0.25, 0.25, 4.0])]])   # velocity unknown
        else:
            self.x, self.P = z.copy(), R.copy()

    def predict(self):
        if self.x is None:
            return
        self.x = self.F @ self.x
        self.x[3] = wrap_quarter(self.x[3])
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas: CubeMeasurement) -> bool:
        z = np.array([*meas.pos, meas.yaw])
        self.last_z = z
        R = self.noise.R(meas.range, self.noise_scale) * self.r_scale
        if self.x is None:
            self._init_from(z, R)
            self.n_updates = 1
            return True
        H = self.H
        y = z - H @ self.x
        y[3] = wrap_quarter(y[3])
        S = H @ self.P @ H.T + R + self.noise.bias_cov * self.noise_scale ** 2
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        if nis > self.GATE:
            self.n_rejected += 1
            self.consec_rejected += 1
            if self.consec_rejected >= self.REINIT_AFTER:   # the track is wrong: drop it and restart
                self._init_from(z, R)
                self.consec_rejected = 0
                self.n_reinit += 1
                return True
            return False
        self.consec_rejected = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = wrap_quarter(self.x[3])
        I = np.eye(self.nx)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T    # Joseph form
        self.n_updates += 1
        return True
