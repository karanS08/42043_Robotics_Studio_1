"""Hardware configuration: one file, one IP address, sensible defaults.

Everything the hardware backend needs lives here so that bring-up is "edit one JSON / pass --ip and go".
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
DEFAULT_PATH = os.path.join(CFG_DIR, "hardware.json")


@dataclass
class HardwareConfig:
    # ---- robot ----
    ip: str = "192.168.1.102"           # UR3e controller address (teach pendant: Settings > System > Network)
    rtde_frequency: float = 100.0
    connect_timeout: float = 3.0        # seconds to wait for the RTDE port before giving up       # Hz; must equal 1 / CTRL_DT so the controller's model matches
    servoj_lookahead: float = 0.1       # s, servoJ smoothing (0.03 - 0.2)
    servoj_gain: float = 300.0          # servoJ proportional gain (100 - 2000)
    headless: bool = False              # True if the robot runs in headless/remote mode

    # ---- gripper ----
    gripper: str = "none"               # "robotiq" | "digital_out" | "none"
    gripper_port: int = 63352           # Robotiq URCap socket port on the controller
    gripper_do: int = 0                 # digital output index when gripper == "digital_out"
    gripper_close_pos: int = 180        # Robotiq 0 (open) .. 255 (closed)
    gripper_open_pos: int = 0
    gripper_speed: int = 120
    gripper_force: int = 80

    # ---- camera ----
    camera: str = "realsense"           # "realsense" | "opencv" | "none"
    camera_serial: str = ""             # RealSense serial; empty = first device found
    camera_index: int = 0               # index when camera == "opencv" (no depth: PBVS needs depth, see docs)
    width: int = 640
    height: int = 480
    fps: int = 30
    align_depth: bool = True

    # ---- safety envelope (ALL enforced before anything is sent to the robot) ----
    max_joint_speed: float = 0.6        # rad/s per joint on the commanded reference
    max_step: float = 0.01              # rad: largest change in a single command (0.01 rad @100 Hz = 1 rad/s)
    ws_x: Tuple[float, float] = (0.10, 0.34)    # TCP box in the robot base frame [m]
    ws_y: Tuple[float, float] = (-0.30, 0.30)
    ws_z: Tuple[float, float] = (0.005, 0.45)   # never below the table
    max_tracking_error: float = 0.10    # rad: abort if the robot falls this far behind the reference
    max_runtime: float = 30.0           # s: hard stop
    home_q: Tuple[float, ...] = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    move_speed: float = 0.4             # rad/s for the initial moveJ to home
    move_accel: float = 0.8

    # ---- identified inner servo model (filled by scripts/hw_identify_servo.py) ----
    servo_kp: Optional[list] = None
    servo_kd: Optional[list] = None

    @staticmethod
    def load(path: Optional[str] = None, ip: Optional[str] = None) -> "HardwareConfig":
        path = path or DEFAULT_PATH
        if os.path.exists(path):
            d = json.load(open(path))
            # This file is meant to be hand-edited, so an unknown key is a typo, not a crash: name it and
            # carry on with the default rather than dying on TypeError in front of the robot.
            known = {k: v for k, v in d.items() if k in HardwareConfig.__dataclass_fields__}
            for k in d.keys() - known.keys():
                print(f"WARNING: {path}: ignoring unknown setting {k!r} (typo? it has no effect)")
            cfg = HardwareConfig(**known)
        else:
            cfg = HardwareConfig()
        if ip:
            cfg.ip = ip
        return cfg

    def save(self, path: Optional[str] = None):
        path = path or DEFAULT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(asdict(self), open(path, "w"), indent=1)
        return path
