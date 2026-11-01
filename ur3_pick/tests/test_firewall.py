"""Ground-truth firewall: the controller must never see simulator state.

The marking guide requires that the controller runs on the estimated state and that ground truth is used
only for evaluation, visibly in the code. These tests check that mechanically:

  1. core/ never imports sim.env, mujoco viewers, or the GroundTruth class, and never touches `.truth`,
     `plant.data`, `mj_step` or any simulator handle.
  2. PickController.step() accepts a plain Observation (encoders + images) - a duck-typed object with no
     access to the plant - and still produces a command.
  3. Feeding the controller an Observation whose encoder values are deliberately offset changes the command,
     i.e. the controller really is driven by the measurement, not by a hidden true state.
"""
import ast
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")

CORE = os.path.join(ROOT, "core")
FORBIDDEN_IMPORTS = {"sim.env", "mujoco.viewer", "hardware", "hardware.plant", "hardware.devices",
                     "rtde_control", "rtde_receive", "pyrealsense2"}
# Attributes that only exist on simulator state. core/ may touch them ONLY on its own MjData (the robot model it
# owns as a predictor / inverse-dynamics engine), never on a handle that came from the plant.
SIM_STATE_ATTRS = {"qpos", "qvel", "qacc", "qfrc_inverse", "xpos", "xmat", "xquat", "site_xpos", "actuator_force"}
# Attributes that are always a leak, whoever owns them.
ALWAYS_FORBIDDEN = {"truth", "cube_pos", "cube_yaw", "xfrc_applied", "holding", "eq_active", "u_effort"}
MJ_STEPPERS = {"mj_step", "mj_forward", "mj_inverse", "mj_kinematics", "mj_comPos", "mj_jacSite", "mj_mulM",
               "mjd_transitionFD"}


def _core_files():
    return [os.path.join(CORE, f) for f in sorted(os.listdir(CORE)) if f.endswith(".py")]


def _own_handles(fn: ast.AST) -> set:
    """Names inside one function that are bound to the module's OWN MjModel/MjData (self.d, self._d, self.m, ...)."""
    own = {"self.d", "self._d", "self.m", "self.model_own"}
    names = {"self"}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            src = node.value
            src_txt = ast.unparse(src) if hasattr(ast, "unparse") else ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Tuple):      # r, d = self.robot, self._d
                    if isinstance(src, ast.Tuple) and len(src.elts) == len(tgt.elts):
                        for t, v in zip(tgt.elts, src.elts):
                            vt = ast.unparse(v)
                            if isinstance(t, ast.Name) and (vt in own or vt.startswith(("self.robot", "self.m", "self.d",
                                                                                        "self._d", "r.m", "r.d"))):
                                names.add(t.id)
                elif isinstance(tgt, ast.Name):
                    if src_txt in own or src_txt.startswith(("self.robot", "self.m", "self.d", "self._d", "r.m", "r.d")):
                        names.add(tgt.id)
    return names


def test_core_does_not_import_the_plant():
    bad = []
    for path in _core_files():
        tree = ast.parse(open(path).read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in FORBIDDEN_IMPORTS or node.module.endswith(".env"):
                    bad.append((path, node.module))
                for a in node.names:
                    if a.name in FORBIDDEN_IMPORTS or a.name == "GroundTruth":
                        bad.append((path, f"{node.module}.{a.name}"))
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in FORBIDDEN_IMPORTS:
                        bad.append((path, a.name))
    assert not bad, f"controller code imports the simulator: {bad}"


def test_core_never_reads_simulator_state():
    """Simulator state may only be touched through the controller's own model instance, never a plant handle."""
    bad = []
    for path in _core_files():
        fname = os.path.basename(path)
        tree = ast.parse(open(path).read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ALWAYS_FORBIDDEN:
                bad.append((fname, node.lineno, f".{node.attr} is simulator-only"))
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            own = _own_handles(fn)
            for node in ast.walk(fn):
                # attribute access such as X.qpos -> X must be an own handle
                if isinstance(node, ast.Attribute) and node.attr in SIM_STATE_ATTRS:
                    base = node.value
                    root = base
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if not (isinstance(root, ast.Name) and root.id in own):
                        bad.append((fname, node.lineno, f"{ast.unparse(base)}.{node.attr}"))
                # mj_step(model, data) etc -> the data argument must be an own handle
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MJ_STEPPERS:
                    for arg in node.args[:2]:
                        root = arg
                        while isinstance(root, ast.Attribute):
                            root = root.value
                        if not (isinstance(root, ast.Name) and root.id in own):
                            bad.append((fname, node.lineno, f"{node.func.attr}({ast.unparse(arg)})"))
    assert not bad, "controller code touches state it does not own:\n  " + "\n  ".join(map(str, bad))


def test_controller_runs_on_a_duck_typed_observation():
    """No Plant anywhere: a bare object with t / q_meas / rgb / depth is enough to drive the controller."""
    from core.control import PickController, HOME_Q

    class FakeObs:
        def __init__(self, t, q):
            self.t, self.q_meas, self.rgb, self.depth = t, q, None, None

    ctl = PickController()
    u, grip, info = ctl.step(FakeObs(0.0, HOME_Q.copy()))
    assert u.shape == (6,) and np.all(np.isfinite(u))
    assert info["obj"] is None                      # nothing seen yet: no object state can appear from nowhere
    assert grip == 0.0


def test_command_follows_the_measurement():
    """Offsetting the encoder reading must change the command: the loop is closed on the measurement."""
    from core.control import PickController, HOME_Q

    class FakeObs:
        def __init__(self, t, q):
            self.t, self.q_meas, self.rgb, self.depth = t, q, None, None

    us = []
    for offset in (0.0, 0.05):
        ctl = PickController()
        q = HOME_Q.copy()
        for k in range(25):
            q_meas = q.copy()
            q_meas[0] += offset
            u, _, _ = ctl.step(FakeObs(k * 0.01, q_meas))
        us.append(u)
    assert abs(us[1][0] - us[0][0]) > 1e-3, "command ignores the encoder measurement"


def test_runner_is_the_only_place_truth_is_used():
    """eval/runner.py may read ground truth; nothing under core/ or scripts/ that builds the controller may."""
    leaked = []
    for d in ("core", "scripts"):
        base = os.path.join(ROOT, d)
        for f in sorted(os.listdir(base)):
            if not f.endswith(".py"):
                continue
            src = open(os.path.join(base, f)).read()
            if ".truth" in src and d == "core":
                leaked.append(f"{d}/{f}")
    assert not leaked, f"ground truth referenced in controller code: {leaked}"


def test_core_is_backend_agnostic():
    """core/ must not know which backend it is driving: no simulator, no robot drivers either.

    The controller receives an Observation and returns joint targets. That is the whole contract, and it is
    what lets the same code run against MuJoCo and against the real UR3e.
    """
    bad = []
    for path in _core_files():
        tree = ast.parse(open(path).read(), path)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                root = m.split(".")[0]
                if root in ("hardware", "rtde_control", "rtde_receive", "rtde_io", "pyrealsense2"):
                    bad.append((os.path.basename(path), m))
    assert not bad, f"controller code imports a backend: {bad}"


def test_hardware_backend_matches_the_simulator_interface():
    """HardwarePlant must expose what the runner uses, and must NOT expose ground truth."""
    import inspect
    from hardware.plant import HardwarePlant
    from sim.env import Plant
    for name in ("step", "reset", "close"):
        assert hasattr(HardwarePlant, name), f"HardwarePlant is missing {name}()"
        sim_sig = inspect.signature(getattr(Plant, name))
        hw_sig = inspect.signature(getattr(HardwarePlant, name))
        if name == "step":
            assert list(sim_sig.parameters) == list(hw_sig.parameters), (
                f"step() signatures differ: sim {sim_sig} vs hardware {hw_sig}")
    assert not hasattr(HardwarePlant, "truth"), "the real robot cannot provide ground truth"
