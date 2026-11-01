"""Frame diagram for Part B: the chain from a camera pixel to the quantity the controller acts on.

Left  : annotated external render with the frames drawn on the robot.
Right : the transform chain, showing which link is measured, which is calibrated, and which is estimated.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sim import scene as S
from sim.env import Plant, PlantConfig
from core.kinematics import RobotModel
from core.sensing import Calibration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOXES = [
    ("pixel  $(u,v,d)$", "RGB-D image", "#dbeafe"),
    ("camera  $p^{cam}$", r"$K^{-1}$, calibrated" "\n" r"$\sigma_d = s_0 + s_2 z^2$", "#dbeafe"),
    ("tool  $p^{tool}$", r"$T^{tool}_{cam}$, hand-eye" "\n" "calibrated, 1.0 mm / 0.16$^\\circ$", "#fde68a"),
    ("base  $p^{base}$", r"$T^{base}_{tool}(\hat q)$, FK of the" "\n" "EKF estimate (not truth)", "#bbf7d0"),
    ("object KF  $\\hat x_{obj}$", r"$[x,y,z,\psi]$, static model," "\n" "gated, range-dependent $R$", "#bbf7d0"),
    ("goal  $p_{ref}$", "phase logic + 2nd-order\nreference filter", "#e9d5ff"),
    ("joints  $q_{ref}$", "damped least-squares IK", "#e9d5ff"),
    ("command  $u$", r"$u_{ff}(q_{ref}) - K(\hat x - x_{ref})$" "\n" "LQR on the estimate", "#fecaca"),
]


def project(free, p_world, W, H, fovy_deg):
    """Pixel (as an axes fraction) of a world point under MuJoCo's free camera convention."""
    az, el = np.radians(free["azimuth"]), np.radians(free["elevation"])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = np.asarray(free["lookat"], float) - free["distance"] * fwd
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    v = np.asarray(p_world, float) - eye
    zc = float(v @ fwd)
    f = (H / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    u_px = W / 2.0 + f * float(v @ right) / zc
    v_px = H / 2.0 - f * float(v @ up) / zc
    return u_px / W, 1.0 - v_px / H


def main():
    plant = Plant(PlantConfig(cube_xy=(0.24, -0.10), cube_yaw=0.4))
    robot = RobotModel()
    _, Rh = robot.fk(S.HOME_Q)
    q, _ = robot.ik(S.HOME_Q, np.array([0.24, -0.02, 0.17]), Rh, iters=200)
    u = robot.servo_equilibrium_u(q)
    for _ in range(400):
        plant.step(u)
    free_cam = dict(lookat=[0.21, -0.05, 0.10], distance=0.72, azimuth=128, elevation=-20)
    ext = plant.render_view(free=free_cam, width=780, height=640)
    cam = plant.render_view("wrist_cam", 320, 240)

    fig = plt.figure(figsize=(13.5, 5.6), dpi=140)
    axim = fig.add_axes([0.005, 0.02, 0.40, 0.95]); axim.imshow(ext); axim.axis("off")
    axim.set_title("UR3e with eye-in-hand RGB-D camera", fontsize=9)
    fovy = float(plant.model.vis.global_.fovy)
    tcp_w = plant.data.site_xpos[plant.tcp_id]
    cam_w = plant.data.cam_xpos[plant.cam_id]
    cube_w = plant.data.xpos[plant.cube_id]
    anchors = [("base frame", (0.0, 0.0, 0.05), (0.04, 0.62)),
               ("tool / TCP", tcp_w, (0.74, 0.44)),
               ("wrist camera", cam_w, (0.70, 0.92)),
               ("object", cube_w, (0.74, 0.13))]
    for txt, pw, xytext in anchors:
        xy = project(free_cam, pw, 780, 640, fovy)
        axim.annotate(txt, xy=xy, xytext=xytext, xycoords="axes fraction", textcoords="axes fraction",
                      fontsize=8, color="w", weight="bold",
                      arrowprops=dict(arrowstyle="->", color="w", lw=1.2),
                      bbox=dict(boxstyle="round,pad=0.25", fc="#00000099", ec="none"))
    axc = fig.add_axes([0.055, 0.045, 0.16, 0.21]); axc.imshow(cam); axc.set_xticks([]); axc.set_yticks([])
    for sp in axc.spines.values():
        sp.set_color("w"); sp.set_linewidth(1.5)
    axc.set_xlabel("wrist camera", fontsize=7, color="w", labelpad=1)

    ax = fig.add_axes([0.43, 0.02, 0.56, 0.95]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(BOXES); h = 0.92 / n
    for i, (title, sub, col) in enumerate(BOXES):
        y = 0.95 - (i + 1) * h
        ax.add_patch(FancyBboxPatch((0.06, y + 0.012), 0.88, h - 0.028, boxstyle="round,pad=0.008",
                                    fc=col, ec="#334155", lw=0.9))
        ax.text(0.10, y + h / 2 + 0.012, title, fontsize=9, va="center", weight="bold")
        ax.text(0.47, y + h / 2 + 0.012, sub, fontsize=7.2, va="center", color="#1f2937")
        if i:
            ax.add_patch(FancyArrowPatch((0.5, y + h - 0.012), (0.5, y + h + 0.004),
                                         arrowstyle="-|>", mutation_scale=9, color="#334155", lw=1.1))
    ax.text(0.5, 0.985, "Measurement $\\rightarrow$ command chain", fontsize=10, ha="center", weight="bold")
    ax.text(0.06, 0.012, "blue = sensor model   amber = calibrated offline   green = estimated online   "
                         "purple = task/reference   red = control",
            fontsize=6.8, color="#334155")
    fig.savefig(os.path.join(ROOT, "results", "fig_frames.png"))
    print("wrote results/fig_frames.png")


if __name__ == "__main__":
    main()
