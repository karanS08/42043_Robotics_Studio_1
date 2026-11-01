"""Device drivers: UR arm over RTDE, RealSense RGB-D camera, Robotiq / digital-output gripper.

All imports of the vendor libraries are lazy, so the simulation side never needs them installed.
Each class exposes the smallest interface the pipeline actually uses, and each has a `connect()` that
fails loudly with an actionable message rather than half-working.
"""
from __future__ import annotations
import socket
import time
from typing import Optional, Tuple
import numpy as np

NJ = 6
RTDE_PORT = 30004          # RTDE real-time port on the UR controller


# --------------------------------------------------------------------------- arm
class URArm:
    """UR3e over RTDE. Commands are joint position targets, exactly what the controller already emits."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ctrl = None
        self.recv = None
        self.io = None
        self.dt = 1.0 / cfg.rtde_frequency

    def connect(self, control: bool = True):
        """control=False opens the receive side only - a dry run can then read the robot but not command it."""
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
            from rtde_io import RTDEIOInterface
        except ImportError as e:
            raise ImportError(
                "ur_rtde is not installed. Install the hardware extras:\n"
                "    .venv/bin/pip install -r requirements-hw.txt") from e
        ip = self.cfg.ip
        # RTDE blocks and retries indefinitely if nothing is listening, so a mistyped IP would look like a
        # hang rather than an error. Check the port first and fail fast with something actionable.
        try:
            with socket.create_connection((ip, RTDE_PORT), timeout=self.cfg.connect_timeout):
                pass
        except OSError as e:
            raise ConnectionError(
                f"nothing is listening on {ip}:{RTDE_PORT} (RTDE) - {type(e).__name__}.\n"
                f"    - is {ip} the robot's address? pendant: Settings -> System -> Network\n"
                f"    - is this PC on the same subnet, and the cable in?\n"
                f"    - is the robot powered on with the brakes released?") from e
        self.recv = RTDEReceiveInterface(ip, self.cfg.rtde_frequency)
        if not control:
            return self
        self.ctrl = RTDEControlInterface(ip, self.cfg.rtde_frequency)
        try:
            self.io = RTDEIOInterface(ip)
        except Exception:
            self.io = None               # optional: only needed for a digital-output gripper
        if not self.ctrl.isConnected():
            raise ConnectionError(f"RTDE control did not connect to {ip}")
        return self

    # ---- state ----
    def q(self) -> np.ndarray:
        return np.asarray(self.recv.getActualQ(), float)

    def qd(self) -> np.ndarray:
        return np.asarray(self.recv.getActualQd(), float)

    def tcp_pose(self) -> np.ndarray:
        return np.asarray(self.recv.getActualTCPPose(), float)

    def robot_mode(self) -> int:
        return int(self.recv.getRobotMode())

    def safety_mode(self) -> int:
        return int(self.recv.getSafetyMode())

    # ---- motion ----
    def move_j(self, q, speed=None, accel=None):
        """Blocking point-to-point move. Used only to go to the start pose."""
        self.ctrl.moveJ(list(map(float, q)), speed or self.cfg.move_speed, accel or self.cfg.move_accel)

    def servo(self, q, dt: Optional[float] = None):
        """Streaming position target. This is the real-time control entry point."""
        if self.ctrl is None:
            raise RuntimeError("arm connected receive-only; servo() is not available in a dry run")
        dt = dt or self.dt
        self.ctrl.servoJ(list(map(float, q)), 0.0, 0.0, dt,
                         self.cfg.servoj_lookahead, self.cfg.servoj_gain)

    def servo_stop(self):
        try:
            self.ctrl.servoStop()
        except Exception:
            pass

    def stop(self):
        try:
            self.ctrl.stopJ(2.0)
        except Exception:
            pass

    def close(self):
        self.servo_stop()
        for h in (self.ctrl, self.recv):
            try:
                h and h.disconnect()
            except Exception:
                pass


# --------------------------------------------------------------------------- camera
class RealSenseCamera:
    """RGB-D camera. Returns (rgb uint8 HxWx3, depth float32 HxW in metres, 0 = invalid)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = None
        self.align = None
        self.scale = 1.0
        self.K_device = None

    def connect(self):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise ImportError(
                "pyrealsense2 is not installed. Install the hardware extras:\n"
                "    .venv/bin/pip install -r requirements-hw.txt") from e
        c = self.cfg
        self.rs = rs
        conf = rs.config()
        if c.camera_serial:
            conf.enable_device(c.camera_serial)
        conf.enable_stream(rs.stream.color, c.width, c.height, rs.format.rgb8, c.fps)
        conf.enable_stream(rs.stream.depth, c.width, c.height, rs.format.z16, c.fps)
        self.pipe = rs.pipeline()
        prof = self.pipe.start(conf)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color) if c.align_depth else None
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K_device = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        for _ in range(10):              # let auto-exposure settle
            self.pipe.wait_for_frames()
        return self

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self.pipe.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            raise RuntimeError("camera returned an empty frame")
        rgb = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.scale
        return rgb, depth

    def close(self):
        try:
            self.pipe and self.pipe.stop()
        except Exception:
            pass


class NoCamera:
    K_device = None

    def connect(self):
        return self

    def read(self):
        raise RuntimeError("no camera configured (set camera in configs/hardware.json)")

    def close(self):
        pass


# --------------------------------------------------------------------------- gripper detection
ROBOTIQ_STA = {0: "reset / not activated", 1: "activating", 2: "activating",
               3: "activated, ready"}


def probe_robotiq(ip: str, port: int = 63352, timeout: float = 2.0) -> dict:
    """Is a Robotiq URCap listening, and what state is its gripper in?

    READ-ONLY. This deliberately never sends `SET ACT 1`: activating a Robotiq makes it run a full
    open/close calibration stroke, which is motion, and a bring-up check must not move anything. Only
    `GET` commands are used, so this is safe to run with the robot powered and someone's hand nearby.
    """
    out = {"kind": "robotiq", "present": False, "detail": ""}
    try:
        with socket.create_connection((ip, port), timeout=timeout) as sk:
            sk.settimeout(timeout)
            sk.sendall(b"GET STA\n")
            sta = sk.recv(256).decode(errors="replace").strip()
            sk.sendall(b"GET POS\n")
            pos = sk.recv(256).decode(errors="replace").strip()
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        out["detail"] = f"no URCap socket on port {port} ({type(e).__name__})"
        return out
    if not sta.startswith("STA"):
        # something answered, but it does not speak the URCap protocol
        out["detail"] = f"port {port} open but replied {sta!r}, not 'STA <n>'"
        return out
    try:
        n = int(sta.split()[1])
    except (IndexError, ValueError):
        out["detail"] = f"unparseable status {sta!r}"
        return out
    out["present"] = True
    out["sta"] = n
    out["detail"] = f"URCap on {port}, status {n} = {ROBOTIQ_STA.get(n, '?')}, {pos}"
    if n != 3:
        out["detail"] += "  -> not activated yet; connect() will activate it (the jaws will stroke once)"
    return out


def probe_digital_out(arm) -> dict:
    """Is the RTDE IO interface usable for a digital-output gripper? Read-only."""
    out = {"kind": "digital_out", "present": False, "detail": ""}
    if getattr(arm, "io", None) is None:
        out["detail"] = ("RTDE IO interface not open (it is only opened by connect(control=True); "
                         "a dry run cannot test this)")
        return out
    try:
        bits = arm.recv.getActualDigitalOutputBits()
        out["present"] = True
        out["detail"] = f"RTDE IO open, digital output bits = 0b{int(bits):08b}"
    except Exception as e:
        out["detail"] = f"RTDE IO open but reading output bits failed: {type(e).__name__}: {e}"
    return out


def detect_gripper(cfg, arm) -> Tuple[str, list]:
    """Best guess at the gripper fitted to this robot, plus every probe result for the report.

    Returns ("robotiq" | "digital_out" | "none", probes). A Robotiq URCap answering on its socket is
    conclusive, so it wins. A usable digital output is not evidence that a gripper is wired to it - every
    UR has digital outputs - so it is reported but never chosen automatically.
    """
    probes = [probe_robotiq(cfg.ip, cfg.gripper_port), probe_digital_out(arm)]
    if probes[0]["present"]:
        return "robotiq", probes
    return "none", probes


# --------------------------------------------------------------------------- gripper
class RobotiqGripper:
    """Robotiq 2F via the URCap socket on the robot controller (port 63352)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sock: Optional[socket.socket] = None

    def _cmd(self, s: str) -> str:
        self.sock.sendall((s + "\n").encode())
        return self.sock.recv(1024).decode().strip()

    def connect(self):
        self.sock = socket.create_connection((self.cfg.ip, self.cfg.gripper_port), timeout=2.0)
        self._cmd("SET ACT 1")
        time.sleep(0.5)
        self._cmd(f"SET SPE {self.cfg.gripper_speed}")
        self._cmd(f"SET FOR {self.cfg.gripper_force}")
        self._cmd("SET GTO 1")
        return self

    def set(self, closed: bool):
        pos = self.cfg.gripper_close_pos if closed else self.cfg.gripper_open_pos
        self._cmd(f"SET POS {pos}")
        self._cmd("SET GTO 1")

    def close(self):
        try:
            self.sock and self.sock.close()
        except Exception:
            pass


class DigitalOutGripper:
    """Any gripper driven by a single digital output on the controller."""

    def __init__(self, cfg, arm):
        self.cfg, self.arm = cfg, arm

    def connect(self):
        if self.arm.io is None:
            raise RuntimeError("RTDE IO interface unavailable; cannot drive a digital-output gripper")
        return self

    def set(self, closed: bool):
        self.arm.io.setStandardDigitalOut(self.cfg.gripper_do, bool(closed))

    def close(self):
        pass


class NoGripper:
    def connect(self):
        return self

    def set(self, closed: bool):
        pass

    def close(self):
        pass


def make_gripper(cfg, arm):
    kinds = {"robotiq": lambda: RobotiqGripper(cfg),
             "digital_out": lambda: DigitalOutGripper(cfg, arm),
             "none": lambda: NoGripper()}
    if cfg.gripper not in kinds:
        raise ValueError(f"configs/hardware.json: gripper={cfg.gripper!r} is not one of "
                         f"{sorted(kinds)}")
    return kinds[cfg.gripper]()


def make_camera(cfg):
    kinds = {"realsense": lambda: RealSenseCamera(cfg),
             "none": lambda: NoCamera()}
    if cfg.camera not in kinds:
        raise ValueError(f"configs/hardware.json: camera={cfg.camera!r} is not one of {sorted(kinds)}. "
                         f"For any other camera, add a class with read() -> (rgb, depth_m); see "
                         f"docs/hardware_bringup.md stage 0.")
    return kinds[cfg.camera]()
