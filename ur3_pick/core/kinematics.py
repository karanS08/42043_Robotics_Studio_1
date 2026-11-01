"""Robot model used by the controller/estimator: FK, Jacobian, damped-least-squares IK, dynamics Jacobians.

This is the *known robot model* (kinematic + inertial parameters from the MJCF), not simulator state.
It owns its own MjModel/MjData so it can be queried at arbitrary (estimated) configurations.
"""
from __future__ import annotations
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

from sim import scene as S

NJ = 6


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


_ROBOT_XML_CACHE: dict = {}      # compiled robot MjModel per process (compiling it per trial leaks ~35 MB)


class RobotModel:
    """The controller's own copy of the robot: kinematics, dynamics and the servo model. Never the plant."""

    def __init__(self, payload_kg: float = 0.0):
        if "m" not in _ROBOT_XML_CACHE:
            m = S.load_model(with_cube=False)
            _ROBOT_XML_CACHE["m"] = m
            _ROBOT_XML_CACHE["mass"] = m.body_mass.copy()
        self.m = _ROBOT_XML_CACHE["m"]
        self.m.body_mass[:] = _ROBOT_XML_CACHE["mass"]
        if payload_kg:
            self.m.body_mass[self.m.body("wrist_3_link").id] += payload_kg
        self.d = mj.MjData(self.m)
        self.tcp = self.m.site("tcp").id
        names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in names])
        self.vadr = np.array([self.m.joint(n).dofadr[0] for n in names])
        self.kp = np.array([self.m.actuator_gainprm[i, 0] for i in range(NJ)])
        self.kd = np.array([-self.m.actuator_biasprm[i, 2] for i in range(NJ)])
        self.jnt_lo = self.m.jnt_range[:NJ, 0].copy()
        self.jnt_hi = self.m.jnt_range[:NJ, 1].copy()

    # ---------------------------------------------------------------- kinematics
    def fk(self, q: np.ndarray):
        """TCP position (3,) and rotation (3,3) in the base frame."""
        self.d.qpos[self.qadr] = q
        mj.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def fk_tool(self, q: np.ndarray):
        """Tool (flange) frame = TCP frame shifted back by TCP_Z along its z axis."""
        p, R = self.fk(q)
        return p - R[:, 2] * S.TCP_Z, R

    def jac(self, q: np.ndarray) -> np.ndarray:
        self.d.qpos[self.qadr] = q
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mj.mj_jacSite(self.m, self.d, jp, jr, self.tcp)
        return np.vstack([jp[:, self.vadr], jr[:, self.vadr]])

    def ik(self, q0: np.ndarray, p_des: np.ndarray, R_des: np.ndarray, iters: int = 25,
           damping: float = 0.02, tol=(1e-5, 1e-4)):
        """Damped least squares IK warm-started from q0. Returns (q, converged)."""
        q = q0.copy()
        for _ in range(iters):
            p, R = self.fk(q)
            e = np.concatenate([p_des - p, so3_log(R_des @ R.T)])
            if np.linalg.norm(e[:3]) < tol[0] and np.linalg.norm(e[3:]) < tol[1]:
                return q, True
            J = self.jac(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.jnt_lo, self.jnt_hi)
        return q, False

    # ---------------------------------------------------------------- dynamics
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        self.d.qpos[self.qadr] = q
        mj.mj_forward(self.m, self.d)
        nv = self.m.nv
        M = np.zeros((nv, nv))
        e, col = np.zeros(nv), np.zeros(nv)
        for i in range(nv):               # M e_i (version-independent; includes armature)
            e[:] = 0; e[i] = 1
            mj.mj_mulM(self.m, self.d, col, e)
            M[:, i] = col
        return M[np.ix_(self.vadr, self.vadr)]

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        self.d.qpos[self.qadr] = q
        self.d.qvel[:] = 0
        mj.mj_forward(self.m, self.d)
        return self.d.qfrc_bias[self.vadr].copy()   # = gravity at zero velocity

    def servo_equilibrium_u(self, q_ref: np.ndarray) -> np.ndarray:
        """Position target that holds q_ref statically: kp (u - q) = g(q)  =>  u = q + g / kp."""
        return q_ref + self.gravity_torque(q_ref) / self.kp

    def linear_model(self, q: np.ndarray):
        """Continuous-time linearisation about (q, qd = 0) with the inner servo closed.

        M(q) qdd + g(q) = Kp (u - q) - Kd qd   =>   with dx = [dq, dqd], du = u - u_eq(q):
            A = [[0, I], [-M^-1 (Kp + dg/dq), -M^-1 Kd]],   B = [[0], [M^-1 Kp]]
        (Coriolis terms vanish at qd = 0; torque saturation is not part of the linear model.)
        """
        M = self.mass_matrix(q)
        G = np.zeros((NJ, NJ)); eps = 1e-5
        for i in range(NJ):
            dq = np.zeros(NJ); dq[i] = eps
            G[:, i] = (self.gravity_torque(q + dq) - self.gravity_torque(q - dq)) / (2 * eps)
        Mi = np.linalg.inv(M)
        Kp, Kd = np.diag(self.kp), np.diag(self.kd)
        A = np.block([[np.zeros((NJ, NJ)), np.eye(NJ)], [-Mi @ (Kp + G), -Mi @ Kd]])
        B = np.vstack([np.zeros((NJ, NJ)), Mi @ Kp])
        return A, B

    def discrete_model(self, q: np.ndarray, dt: float):
        """Exact zero-order-hold discretisation of linear_model."""
        from scipy.linalg import expm
        A, B = self.linear_model(q)
        n = A.shape[0]
        aug = np.zeros((n + NJ, n + NJ)); aug[:n, :n] = A; aug[:n, n:] = B
        E = expm(aug * dt)
        return E[:n, :n], E[:n, n:]
