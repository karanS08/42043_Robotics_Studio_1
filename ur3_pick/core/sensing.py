"""Sensing: camera model, frame chain, cube detection and depth-based localisation.

Frame chain (all rigid transforms, OpenCV camera convention x right / y down / z forward):
    p_base = T_base_tool(q_hat) * T_tool_cam * p_cam
T_tool_cam and K come from calibration (scripts/calibrate.py -> configs/calibration.json), never from the simulator.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

from .kinematics import RobotModel

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
CUBE_HALF = 0.02       # known object size (task knowledge)


@dataclass
class Calibration:
    K: np.ndarray            # 3x3 intrinsics
    T_tool_cam: np.ndarray   # 4x4, camera pose in tool frame (OpenCV convention)

    @staticmethod
    def load(path: Optional[str] = None) -> "Calibration":
        path = path or os.path.join(CFG_DIR, "calibration.json")
        d = json.load(open(path))
        return Calibration(np.array(d["K"]), np.array(d["T_tool_cam"]))

    def perturbed(self, dp_m: float, drot_deg: float, rng: np.random.Generator) -> "Calibration":
        """Calibration with an injected hand-eye error (robustness study)."""
        from scipy.spatial.transform import Rotation as R
        dirv = rng.normal(size=3); dirv /= np.linalg.norm(dirv)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        T = self.T_tool_cam.copy()
        T[:3, 3] += dirv * dp_m
        T[:3, :3] = R.from_rotvec(axis * np.radians(drot_deg)).as_matrix() @ T[:3, :3]
        return Calibration(self.K, T)


@dataclass
class SegmentParams:
    """Colour-segmentation thresholds for the red cube's top face.

    The defaults isolate the cube under the simulator's lighting. Real lab lighting shifts the whole
    histogram - fluorescent tubes desaturate, a bright window clips the red channel - so these live in
    configs/segment.json and are fitted on the real camera by scripts/hw_tune_segment.py. Nothing else
    in the pipeline changes; only which pixels are called "cube".
    """
    r_min: int = 110          # absolute red level
    r_minus_g: int = 60       # redness against green
    r_minus_b: int = 60       # redness against blue
    open_ksize: int = 3       # morphological opening kernel, removes speckle

    @staticmethod
    def load(path: Optional[str] = None) -> "SegmentParams":
        path = path or os.path.join(CFG_DIR, "segment.json")
        if os.path.exists(path):
            d = json.load(open(path))
            return SegmentParams(**{k: v for k, v in d.items()
                                    if k in SegmentParams.__dataclass_fields__})
        return SegmentParams()

    def save(self, path: Optional[str] = None, **extra):
        path = path or os.path.join(CFG_DIR, "segment.json")
        d = dict(r_min=int(self.r_min), r_minus_g=int(self.r_minus_g),
                 r_minus_b=int(self.r_minus_b), open_ksize=int(self.open_ksize))
        d.update(extra)
        json.dump(d, open(path, "w"), indent=2)
        return path


@dataclass
class CubeMeasurement:
    pos: np.ndarray        # (3,) cube centre in base frame
    yaw: float             # yaw modulo 90 deg in (-pi/4, pi/4]
    range: float           # camera -> cube distance [m]
    n_pix: int             # number of top-face pixels used


def wrap_quarter(a: float) -> float:
    """Wrap an angle to (-pi/4, pi/4] (cube symmetry)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def segment_red(rgb: np.ndarray, seg: "SegmentParams") -> np.ndarray:
    """Largest connected red blob in the image, as a boolean mask."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    mask = ((r > seg.r_min) & (r - g > seg.r_minus_g) & (r - b > seg.r_minus_b)).astype(np.uint8)
    k = max(1, int(seg.open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask, bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


class CubeDetector:
    """Colour segmentation + depth back-projection + planar top-face fit."""

    def __init__(self, robot: RobotModel, calib: Calibration, min_pixels: int = 25, z_estimator: str = "median",
                 seg: Optional[SegmentParams] = None):
        # z_estimator: "median" (robust, default) or "p90" (first version; biased upwards under depth noise -
        # kept so the failure analysis in the report can be reproduced)
        self.robot, self.calib, self.min_pixels, self.z_estimator = robot, calib, min_pixels, z_estimator
        self.seg = seg if seg is not None else SegmentParams.load()

    def T_base_cam(self, q: np.ndarray) -> np.ndarray:
        p, R = self.robot.fk_tool(q)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
        return T @ self.calib.T_tool_cam

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        return segment_red(rgb, self.seg)

    def detect(self, rgb: np.ndarray, depth: np.ndarray, q_hat: np.ndarray) -> Optional[CubeMeasurement]:
        mask = self.segment(rgb)
        if mask.sum() < self.min_pixels:
            return None
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            return None                                   # cube cut by the image border -> biased centroid, reject
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # drop depth-edge pixels
        pc = backproject(depth, self.calib.K, mask)
        if len(pc) < self.min_pixels:
            return None
        T = self.T_base_cam(q_hat)
        pb = pc @ T[:3, :3].T + T[:3, 3]
        if self.z_estimator == "p90":
            top = np.percentile(pb[:, 2], 90)
            sel = pb[:, 2] > top - 0.006
            if sel.sum() < self.min_pixels:
                return None
            z_top = float(np.median(pb[sel, 2]))
        else:
            # top face = majority of the visible pixels -> its height is the median; side-face pixels lie well below
            z_top = float(np.median(pb[:, 2]))
            sel = pb[:, 2] > z_top - 0.008
            if sel.sum() < self.min_pixels:
                return None
        ptop = pb[sel]
        c_xy = ptop[:, :2].mean(axis=0)
        (_, _), (_, _), ang = cv2.minAreaRect(ptop[:, :2].astype(np.float32))
        yaw = wrap_quarter(np.radians(ang))
        rng = float(np.mean(pc[sel, 2]))
        return CubeMeasurement(np.array([c_xy[0], c_xy[1], z_top - CUBE_HALF]), yaw, rng, int(sel.sum()))
