"""Safety envelope. Every command to the real robot passes through this; nothing bypasses it.

The checks are deliberately conservative and ordered cheapest-first. Any violation raises SafetyAbort,
which the runner catches to stop the robot (servoStop) before re-raising.
"""
from __future__ import annotations
import numpy as np

NJ = 6


class SafetyAbort(RuntimeError):
    pass


class SafetyLimiter:
    def __init__(self, cfg, robot_model):
        self.cfg = cfg
        self.rm = robot_model
        self.q_prev: np.ndarray | None = None
        self.n_clamped = 0
        self.reasons: list[str] = []

    def reset(self, q0: np.ndarray):
        self.q_prev = np.asarray(q0, float).copy()
        self.n_clamped = 0
        self.reasons = []

    def check_state(self, q_meas: np.ndarray, q_cmd_prev: np.ndarray | None, t: float):
        """Checks on what the robot is actually doing, before computing a new command."""
        c = self.cfg
        if not np.all(np.isfinite(q_meas)):
            raise SafetyAbort("non-finite joint measurement from the robot")
        if t > c.max_runtime:
            raise SafetyAbort(f"runtime exceeded {c.max_runtime} s")
        lo, hi = self.rm.jnt_lo, self.rm.jnt_hi
        if np.any(q_meas < lo - 0.05) or np.any(q_meas > hi + 0.05):
            raise SafetyAbort(f"joint outside limits: {np.round(q_meas, 3)}")
        p, _ = self.rm.fk(q_meas)
        if not (c.ws_x[0] - 0.05 <= p[0] <= c.ws_x[1] + 0.05 and c.ws_y[0] - 0.05 <= p[1] <= c.ws_y[1] + 0.05
                and c.ws_z[0] - 0.05 <= p[2] <= c.ws_z[1] + 0.05):
            raise SafetyAbort(f"TCP left the safe box: {np.round(p, 3)}")
        if q_cmd_prev is not None:
            err = float(np.max(np.abs(q_meas - q_cmd_prev)))
            if err > c.max_tracking_error:
                raise SafetyAbort(f"robot is {err:.3f} rad behind the reference (limit {c.max_tracking_error})")

    def limit_command(self, q_cmd: np.ndarray, dt: float) -> np.ndarray:
        """Clamp a candidate command: joint limits, step size, speed, and TCP box."""
        c = self.cfg
        q = np.asarray(q_cmd, float).copy()
        if not np.all(np.isfinite(q)):
            raise SafetyAbort("controller produced a non-finite command")
        q = np.clip(q, self.rm.jnt_lo, self.rm.jnt_hi)
        if self.q_prev is not None:
            step_cap = min(c.max_step, c.max_joint_speed * dt)
            d = np.clip(q - self.q_prev, -step_cap, step_cap)
            if np.any(np.abs(q - self.q_prev) > step_cap + 1e-12):
                self.n_clamped += 1
            q = self.q_prev + d
        p, _ = self.rm.fk(q)
        inside = (c.ws_x[0] <= p[0] <= c.ws_x[1] and c.ws_y[0] <= p[1] <= c.ws_y[1]
                  and c.ws_z[0] <= p[2] <= c.ws_z[1])
        if not inside:
            # refuse to move further out: hold the previous command rather than guess a projection
            self.n_clamped += 1
            self.reasons.append(f"command TCP outside the box at {np.round(p, 3)}; holding")
            q = self.q_prev.copy() if self.q_prev is not None else q
        self.q_prev = q
        return q
