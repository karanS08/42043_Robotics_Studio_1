"""HardwarePlant: the real UR3e behind exactly the interface the controller already consumes.

    obs = plant.reset()
    while ...:
        u, grip, info = controller.step(obs)
        obs = plant.step(u, grip)

Same `Observation` dataclass as the simulator, same units (rad, metres, seconds). The difference the
controller cannot see, and must not need to: there is no ground truth here. `HardwarePlant` deliberately
has no `.truth`, so any code that reaches for it fails loudly instead of silently reading nothing.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np

from sim.env import Observation, CTRL_DT, CAM_PERIOD
from .config import HardwareConfig
from .devices import URArm, make_camera, make_gripper
from .safety import SafetyLimiter, SafetyAbort

NJ = 6


class HardwarePlant:
    def __init__(self, cfg: HardwareConfig, robot_model, cam_period: int = CAM_PERIOD):
        self.cfg = cfg
        self.arm = URArm(cfg)
        self.cam = make_camera(cfg)
        self.gripper = None
        self.safety = SafetyLimiter(cfg, robot_model)
        self.cam_period = cam_period
        self.k = 0
        self.t0 = 0.0
        self._last_cmd: Optional[np.ndarray] = None
        self._last_obs: Optional[Observation] = None
        self.late_frames = 0
        self.connected = False

    # ------------------------------------------------------------------ lifecycle
    def connect(self, control: bool = True):
        """control=False: sensors only. The control interface is never opened, so nothing can move."""
        self.arm.connect(control=control)
        self.gripper = make_gripper(self.cfg, self.arm).connect() if control else None
        self.cam.connect()
        self.connected = True
        return self

    def go_home(self):
        """Blocking move to the start pose. Point-to-point, not servo: this is the only unguarded motion."""
        self.arm.move_j(self.cfg.home_q)
        time.sleep(0.3)

    def reset(self) -> Observation:
        q = self.arm.q()
        self.safety.reset(q)
        self.k = 0
        self.t0 = time.monotonic()
        self._last_cmd = q.copy()
        self._last_obs = self._observe(0.0, force_camera=True)
        return self._last_obs

    def close(self):
        try:
            self.arm.servo_stop()
        finally:
            for h in (self.gripper, self.cam, self.arm):
                try:
                    h and h.close()
                except Exception:
                    pass
            self.connected = False

    # ------------------------------------------------------------------ stepping
    def _observe(self, t: float, force_camera: bool = False) -> Observation:
        obs = Observation(t=t, q_meas=self.arm.q())
        if force_camera or self.k % self.cam_period == 0:
            try:
                obs.rgb, obs.depth = self.cam.read()
            except Exception as e:                 # a dropped frame must not kill the run
                self.late_frames += 1
                if self.late_frames > 50:
                    raise SafetyAbort(f"camera failed repeatedly: {e}")
        return obs

    def step(self, u: np.ndarray, grip: float = 0.0) -> Observation:
        t = time.monotonic() - self.t0
        self.safety.check_state(self._last_obs.q_meas, self._last_cmd, t)
        q_cmd = self.safety.limit_command(u, CTRL_DT)
        self.arm.servo(q_cmd)
        self._last_cmd = q_cmd
        if self.gripper is not None:
            self.gripper.set(grip > 0.5)
        # hold the control period
        target = self.t0 + (self.k + 1) * CTRL_DT
        slack = target - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        else:
            self.late_frames += 0
        self.k += 1
        self._last_obs = self._observe(time.monotonic() - self.t0)
        return self._last_obs
