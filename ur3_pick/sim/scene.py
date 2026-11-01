"""Scene builder: UR3e (third-party MJCF) + table + cube + wrist RGB-D camera + two-finger gripper.

The UR3e XML is textually extended (camera, fingers, TCP site go inside wrist_3_link) so the
third-party file stays untouched. All numbers that define the hand-eye transform live here;
the *controller* never reads them - it uses the calibrated estimate from configs/handeye.json.
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "model")

# Rotation of the attachment site (tool frame) expressed in the wrist_3_link frame.
# Columns are the tool x, y, z axes in wrist_3 coordinates (verified numerically from ur3e.xml).
R_W3_TOOL = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]])
TOOL_POS_W3 = np.array([0.0, 0.09215, 0.0])  # attachment_site origin in wrist_3_link

TCP_Z = 0.15                    # tool-centre-point distance along tool z from the flange [m]
CUBE_HALF = 0.02                # cube half side [m]
TABLE_Z = 0.0                   # table top height in base frame
IMG_W, IMG_H, FOVY_DEG = 320, 240, 70.0
HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# Eye-in-hand camera pose in the TOOL frame (ground truth of the simulator).
CAM_POS_TOOL = np.array([0.0, 0.06, 0.02])
CAM_TARGET_TOOL = np.array([0.0, 0.0, 0.20])   # point the camera looks at (tool frame)


def cam_rot_tool() -> np.ndarray:
    """Rotation of the MuJoCo camera frame (x right, y up, looks along -z) in the tool frame."""
    zc = -(CAM_TARGET_TOOL - CAM_POS_TOOL)
    zc /= np.linalg.norm(zc)
    up = np.array([0.0, 1.0, 0.0]) if abs(zc[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    xc = np.cross(up, zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return np.stack([xc, yc, zc], axis=1)


def true_T_tool_cam_opencv() -> np.ndarray:
    """Ground-truth hand-eye transform tool -> camera in the OpenCV convention (x right, y down, z fwd)."""
    Rm = cam_rot_tool()
    Rcv = Rm @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4); T[:3, :3] = Rcv; T[:3, 3] = CAM_POS_TOOL
    return T


def intrinsics() -> np.ndarray:
    f = (IMG_H / 2.0) / np.tan(np.radians(FOVY_DEG) / 2.0)
    return np.array([[f, 0, 159.5], [0, f, 119.5], [0, 0, 1.0]])


def _w3(v_tool):  # tool-frame vector -> wrist_3_link frame
    return R_W3_TOOL @ np.asarray(v_tool)


def _w3p(p_tool):  # tool-frame point -> wrist_3_link frame
    return TOOL_POS_W3 + R_W3_TOOL @ np.asarray(p_tool)


def _fmt(v):
    return " ".join(f"{x:.6f}" for x in v)


def _wrist_additions(finger_open: float = 0.04) -> str:
    Rm = cam_rot_tool()
    xaxis, yaxis = _w3(Rm[:, 0]), _w3(Rm[:, 1])
    cam = (f'<camera name="wrist_cam" pos="{_fmt(_w3p(CAM_POS_TOOL))}" '
           f'xyaxes="{_fmt(xaxis)} {_fmt(yaxis)}" fovy="{FOVY_DEG}"/>')
    from scipy.spatial.transform import Rotation as _R
    qx = _R.from_matrix(R_W3_TOOL).as_quat()          # x, y, z, w
    qwxyz = [qx[3], qx[0], qx[1], qx[2]]
    tcp = (f'<site name="tcp" pos="{_fmt(_w3p([0, 0, TCP_Z]))}" quat="{_fmt(qwxyz)}" '
           'size="0.004" rgba="0 0 0 0" group="4"/>')
    fingers = ""
    for i, sgn in enumerate((+1, -1)):
        # finger box: half sizes (x: 0.005, y: 0.01, z: 0.065) centred at tool z = 0.095 (0.03 .. 0.16)
        pos = _w3p([sgn * finger_open, 0, 0.095])
        # box half-size expressed in wrist_3 axes: (x,z,y) swap handled by building the geom in w3 frame
        hs_tool = np.array([0.005, 0.01, 0.065])
        hs_w3 = np.abs(R_W3_TOOL @ hs_tool)
        fingers += (f'<geom name="finger{i}" type="box" pos="{_fmt(pos)}" size="{_fmt(hs_w3)}" '
                    f'rgba="0.15 0.15 0.15 1" mass="0.02" friction="1.5 0.05 0.01" group="1"/>')
    return cam + tcp + fingers


def build_scene_xml(board: bool = False, cube_xy=(0.22, -0.22), cube_yaw=0.0, with_cube=True) -> str:
    """Return a self-contained MJCF string (meshdir relative to MODEL_DIR; load with from_xml_string + assets)."""
    robot = open(os.path.join(MODEL_DIR, "ur3e.xml")).read()
    robot = robot.replace('<mujoco model="ur3e">', '<mujoco model="ur3_pick">')
    robot = robot.replace('<option integrator="implicitfast"/>',
                          '<option integrator="implicitfast" timestep="0.002"/>'
                          '<visual><global offwidth="1280" offheight="960"/>'
                          '<headlight ambient="0.2 0.2 0.2" diffuse="0.15 0.15 0.15" specular="0 0 0"/>'
                          '<map znear="0.02" zfar="3"/></visual>')
    # inject wrist additions just before the closing tag of wrist_3_link
    marker = '<site name="attachment_site"'
    i = robot.index(marker)
    j = robot.index("</body>", i)
    robot = robot[:j] + _wrist_additions() + robot[j:]

    world = (
        '<geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.05" rgba="0.55 0.55 0.58 1" contype="0" conaffinity="0"/>'
        f'<geom name="table" type="box" pos="0.25 0 {TABLE_Z - 0.025}" size="0.45 0.45 0.025" rgba="0.42 0.40 0.36 1" friction="1 0.05 0.01"/>'
    )
    if with_cube:
        qz = np.sin(cube_yaw / 2), np.cos(cube_yaw / 2)
        world += (f'<body name="cube" pos="{cube_xy[0]} {cube_xy[1]} {TABLE_Z + CUBE_HALF}" quat="{qz[1]} 0 0 {qz[0]}">'
                  '<freejoint name="cube_free"/>'
                  f'<geom name="cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" mass="0.05" '
                  'rgba="0.9 0.1 0.1 1" friction="1.2 0.05 0.01"/></body>')
    if board:
        # 8x7 squares (inner corners 7x6; odd x even removes the 180 deg ambiguity), 25 mm squares, lying on the table, black on white
        sq, nx, ny = 0.025, 8, 7
        ox, oy = cube_xy
        world += (f'<geom name="board_bg" type="box" pos="{ox} {oy} {TABLE_Z + 0.0005}" '
                  f'size="{(nx + 2) * sq / 2} {(ny + 2) * sq / 2} 0.0005" rgba="1 1 1 1" contype="0" conaffinity="0"/>')
        for a in range(nx):
            for b in range(ny):
                if (a + b) % 2 == 0:
                    px = ox + (a - nx / 2 + 0.5) * sq
                    py = oy + (b - ny / 2 + 0.5) * sq
                    world += (f'<geom type="box" pos="{px} {py} {TABLE_Z + 0.0011}" size="{sq / 2} {sq / 2} 0.0005" '
                              'rgba="0 0 0 1" contype="0" conaffinity="0"/>')
    world += ('<light name="top" pos="0.3 0 1.5" dir="0 0 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>'
              '<light name="side" pos="-0.5 -0.5 1.0" dir="0.5 0.5 -1" diffuse="0.2 0.2 0.2" castshadow="false"/>')
    robot = robot.replace("</worldbody>", world + "</worldbody>", 1)
    if with_cube:
        weld = ('<equality><weld name="grasp" body1="wrist_3_link" body2="cube" active="false" '
                'solref="0.01 1" solimp="0.95 0.99 0.001"/></equality>')
        robot = robot.replace("</actuator>", "</actuator>" + weld, 1)
    return robot


def load_model(**kw):
    import mujoco
    xml = build_scene_xml(**kw)
    # write next to the model so relative meshdir resolves
    path = os.path.join(MODEL_DIR, "_scene_tmp_%d.xml" % os.getpid())
    with open(path, "w") as f:
        f.write(xml)
    try:
        m = mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)
    return m
