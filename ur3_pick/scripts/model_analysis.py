"""Part A: state-space model of the UR3e (inner servo closed), controllability / observability, validation against MuJoCo.

State x = [q - q0, qd] (12), input u = joint position targets - u_eq (6), output y = encoder angles (6).
    M(q) qdd + c(q, qd) + g(q) = Kp (u - q) - Kd qd
    A = [[0, I], [-M^-1 (Kp + dg/dq), -M^-1 Kd]],  B = [[0], [M^-1 Kp]],  C = [I 0]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj
import control as ct

from sim import scene as S
from core.kinematics import RobotModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NJ = 6
JN = ["sh_pan", "sh_lift", "elbow", "wrist1", "wrist2", "wrist3"]


def simulate_nonlinear(robot, q0, du, T=0.6, dt=0.01):
    """Step the full MuJoCo model from equilibrium with a step du on the position targets."""
    m = robot.m; d = mj.MjData(m)
    d.qpos[robot.qadr] = q0
    u0 = robot.servo_equilibrium_u(q0)
    d.ctrl[:NJ] = u0
    for _ in range(1500):
        mj.mj_step(m, d)                       # settle exactly on the equilibrium
    qeq = d.qpos[robot.qadr].copy()
    d.ctrl[:NJ] = u0 + du
    out = []
    for _ in range(int(T / dt)):
        for _ in range(int(round(dt / m.opt.timestep))):
            mj.mj_step(m, d)
        out.append(np.concatenate([d.qpos[robot.qadr] - qeq, d.qvel[robot.vadr]]))
    return np.array(out)


def main():
    robot = RobotModel()
    _, Rh = robot.fk(S.HOME_Q)
    configs = {"home": S.HOME_Q,
               "scan": robot.ik(S.HOME_Q, np.array([0.22, 0.10, 0.30]), Rh, iters=200)[0],
               "grasp": robot.ik(S.HOME_Q, np.array([0.24, -0.15, 0.02]), Rh, iters=200)[0]}
    C = np.hstack([np.eye(NJ), np.zeros((NJ, NJ))])
    out = {}
    for name, q in configs.items():
        A, B = robot.linear_model(q)
        Ad, Bd = robot.discrete_model(q, 0.01)
        M = robot.mass_matrix(q)
        poles = np.linalg.eigvals(A)
        sv_c = np.linalg.svd(ct.ctrb(Ad, Bd), compute_uv=False)
        sv_o = np.linalg.svd(ct.obsv(Ad, C), compute_uv=False)
        # what if only joint velocities were measured (no encoder angle)?  -> still observable because of the servo stiffness
        sv_o_vel = np.linalg.svd(ct.obsv(Ad, np.hstack([np.zeros((NJ, NJ)), np.eye(NJ)])), compute_uv=False)
        # a single encoder (shoulder pan only)
        sv_o_one = np.linalg.svd(ct.obsv(Ad, C[:1]), compute_uv=False)
        out[name] = dict(
            q=q.tolist(), M_diag=np.diag(M).tolist(), cond_M=float(np.linalg.cond(M)),
            poles_real=np.sort(poles.real).tolist(), poles_imag_max=float(np.abs(poles.imag).max()),
            ctrb_rank=int(np.linalg.matrix_rank(ct.ctrb(Ad, Bd), tol=1e-9)), ctrb_sv_min=float(sv_c[11]), ctrb_sv_max=float(sv_c[0]),
            obsv_rank=int(np.linalg.matrix_rank(ct.obsv(Ad, C), tol=1e-9)), obsv_sv_min=float(sv_o[11]), obsv_sv_max=float(sv_o[0]),
            obsv_rank_velocity_only=int(np.linalg.matrix_rank(ct.obsv(Ad, np.hstack([np.zeros((NJ, NJ)), np.eye(NJ)])), tol=1e-9)),
            obsv_sv_min_velocity_only=float(sv_o_vel[11]),
            obsv_rank_one_encoder=int(np.linalg.matrix_rank(ct.obsv(Ad, C[:1]), tol=1e-9)),
            obsv_sv_min_one_encoder=float(sv_o_one[11]),
            max_abs_eig_Ad=float(np.abs(np.linalg.eigvals(Ad)).max()),
        )
        print(f"[{name}] M diag {np.diag(M).round(3)}  poles re [{poles.real.min():.1f}, {poles.real.max():.1f}]"
              f"  ctrb rank {out[name]['ctrb_rank']}/12 (sv_min {sv_c[11]:.2e})  obsv rank {out[name]['obsv_rank']}/12 (sv_min {sv_o[11]:.2e})"
              f"  obsv(vel only) {out[name]['obsv_rank_velocity_only']}/12  obsv(1 encoder) {out[name]['obsv_rank_one_encoder']}/12")

    # ---------------- validation: linear model vs full nonlinear MuJoCo step responses
    q0 = configs["scan"]
    Ad, Bd = robot.discrete_model(q0, 0.01)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    val = {}
    for amp, ls in ((0.02, "-"), (0.3, "--")):
        errs = []
        for j in range(NJ):
            du = np.zeros(NJ); du[j] = amp
            nl = simulate_nonlinear(robot, q0, du)
            x = np.zeros(12); lin = []
            for _ in range(len(nl)):
                x = Ad @ x + Bd @ du
                lin.append(x.copy())
            lin = np.array(lin)
            errs.append(float(np.sqrt(((lin[:, :6] - nl[:, :6]) ** 2).mean()) / amp))
            ax = axes.flat[j]
            t = np.arange(1, len(nl) + 1) * 0.01
            ax.plot(t, nl[:, j] / amp, "k" + ls, lw=1.6, label=f"MuJoCo, step {amp} rad")
            ax.plot(t, lin[:, j] / amp, "r" + ls, lw=1.0, label=f"linear model, step {amp} rad")
            ax.set_title(JN[j]); ax.set_xlabel("t [s]"); ax.set_ylabel("q / step")
            ax.grid(alpha=0.3)
        val[f"step_{amp}"] = dict(rel_rms_err_per_joint=errs)
        print(f"step {amp} rad: relative RMS error linear vs MuJoCo per joint: {np.round(errs, 4)}")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Linear servo + rigid-body model vs MuJoCo (normalised step responses at the scan pose)")
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results", "model_validation.png"), dpi=130)
    out["validation"] = val
    out["servo"] = dict(kp=robot.kp.tolist(), kd=robot.kd.tolist(),
                        force_limit=robot.m.actuator_forcerange[:NJ, 1].tolist())
    json.dump(out, open(os.path.join(ROOT, "results", "model_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
