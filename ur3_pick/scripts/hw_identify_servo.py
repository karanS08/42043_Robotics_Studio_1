"""Stage 2: identify the real inner servo loop so Part A's model describes the ACTUAL platform.

THE ROBOT MOVES. Small joint steps (default 0.03 rad) about the home pose, one joint at a time.
Stand clear, keep the e-stop in hand, and run scripts/hw_check.py first.

    python scripts/hw_identify_servo.py --ip 192.168.1.102 --amplitude 0.03

For each joint it commands a step through servoJ, records the response, and fits the second-order model the
simulation uses, M_ii qdd = kp (u - q) - kd qd, giving kp and kd per joint. Results are written into
configs/hardware.json (servo_kp / servo_kd) and a comparison figure to results/hw_servo_identification.png.
These numbers are what makes the hardware LQR a model of the real robot rather than of MuJoCo.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from hardware.config import HardwareConfig
from core.kinematics import RobotModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JN = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def step_response(wn, zeta, t, amp):
    """Unit step of a second-order system, scaled by amp."""
    wd = wn * np.sqrt(np.clip(1 - zeta ** 2, 1e-9, None))
    return amp * (1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip")
    ap.add_argument("--amplitude", type=float, default=0.03, help="step size [rad]")
    ap.add_argument("--duration", type=float, default=1.2, help="record window per joint [s]")
    ap.add_argument("--joints", default="0,1,2,3,4,5")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()
    cfg = HardwareConfig.load(ip=a.ip)
    dt = 1.0 / cfg.rtde_frequency

    print(f"\nServo identification on {cfg.ip}.")
    print(f"The robot WILL MOVE: {a.amplitude:.3f} rad steps about the home pose, one joint at a time.")
    if not a.yes and input("Area clear and e-stop in hand? type 'move' to continue: ").strip() != "move":
        print("aborted."); return 1

    from hardware.devices import URArm
    arm = URArm(cfg).connect()
    rm = RobotModel()
    print("moving to home ...")
    arm.move_j(cfg.home_q)
    time.sleep(0.5)

    kp_fit, kd_fit, traces = [], [], {}
    for j in [int(x) for x in a.joints.split(",")]:
        q0 = arm.q()
        target = q0.copy(); target[j] += a.amplitude
        ts, qs = [], []
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t > a.duration:
                break
            arm.servo(target, dt)
            ts.append(t); qs.append(arm.q()[j])
            slack = t0 + len(ts) * dt - time.monotonic()
            if slack > 0:
                time.sleep(slack)
        arm.servo_stop()
        ts, qs = np.array(ts), np.array(qs) - q0[j]
        try:
            (wn, zeta), _ = curve_fit(lambda t, wn, z: step_response(wn, z, t, a.amplitude), ts, qs,
                                      p0=[20.0, 1.0], bounds=([1.0, 0.1], [200.0, 5.0]))
        except Exception as e:
            print(f"  joint {j} ({JN[j]}): fit failed ({e}); skipping")
            continue
        M = rm.mass_matrix(q0)[j, j]
        kp, kd = M * wn ** 2, 2 * zeta * wn * M
        kp_fit.append(kp); kd_fit.append(kd); traces[j] = (ts, qs, wn, zeta)
        print(f"  joint {j} ({JN[j]:13s}) wn {wn:6.1f} rad/s  zeta {zeta:4.2f}  M {M:.3f} -> kp {kp:7.1f}  kd {kd:6.1f}")
        arm.move_j(q0)
        time.sleep(0.3)

    arm.close()
    if len(kp_fit) == 6:
        cfg.servo_kp = [float(x) for x in kp_fit]
        cfg.servo_kd = [float(x) for x in kd_fit]
        print(f"\nwrote {cfg.save()}  (controller can now run with feedforward='gains')")
    else:
        print("\nnot all six joints identified; configs/hardware.json left unchanged")

    if traces:
        n = len(traces)
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3), squeeze=False)
        for ax, (j, (ts, qs, wn, zeta)) in zip(axes.flat, sorted(traces.items())):
            ax.plot(ts, 1e3 * qs, "k", lw=1.4, label="robot")
            ax.plot(ts, 1e3 * step_response(wn, zeta, ts, a.amplitude), "r--", lw=1.0, label="2nd-order fit")
            ax.axhline(1e3 * a.amplitude, color="0.6", lw=0.6)
            ax.set_title(f"{JN[j]}\n$\\omega_n$={wn:.0f}, $\\zeta$={zeta:.2f}", fontsize=8)
            ax.set_xlabel("t [s]"); ax.grid(alpha=0.3)
        axes.flat[0].set_ylabel("step response [mrad]"); axes.flat[0].legend(fontsize=6)
        fig.suptitle("UR3e inner servo loop identified through servoJ", fontsize=9)
        fig.tight_layout()
        out = os.path.join(ROOT, "results", "hw_servo_identification.png")
        fig.savefig(out, dpi=130); print("figure:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
