"""All report figures. Reads results/*.json (Monte Carlo + tuning) and runs a few logged trials for time series.

Figures written to results/:
  fig_estimation_timeseries.png   estimate vs truth with +-2 sigma, joint EKF + object KF (nominal and 4x noise)
  fig_consistency.png             NEES / NIS over time and their chi2 bounds, plus the Q/R sweeps
  fig_tracking.png                joint tracking, Cartesian tracking, control effort: LQR vs servo-only
  fig_lqr_tradeoff.png            tracking error vs effort vs chatter over the LQR weight grid
  fig_robustness.png              success rate + grasp error per condition, with Wilson intervals
  fig_object_kf.png               object estimate error vs raw detection, covariance shrink, outlier rejection
"""
import os, sys, json, glob
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from eval.runner import TrialConfig, run_trial
from core.control import ControllerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
JN = ["sh_pan", "sh_lift", "elbow", "wrist1", "wrist2", "wrist3"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titlesize": 9, "legend.fontsize": 7.5})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ----------------------------------------------------------------- time-series figures
def fig_estimation(logs):
    A = logs["nominal"]["log"]
    t = A["t"]
    fig, ax = plt.subplots(2, 3, figsize=(12, 5.4))
    for j in range(3):          # three most-excited joints
        e = A["q_hat"][:, j] - A["q"][:, j]
        s = 2 * np.sqrt(A["P_q"][:, j])
        ax[0, j].plot(t, e * 1e3, "C0", lw=0.8, label="error")
        ax[0, j].fill_between(t, -s * 1e3, s * 1e3, color="C0", alpha=0.18, label=r"$\pm 2\sigma$")
        ax[0, j].set_title(f"joint angle error, {JN[j]}"); ax[0, j].set_ylabel("mrad")
        ev = A["qd_hat"][:, j] - A["qd"][:, j]
        sv = 2 * np.sqrt(A["P_v"][:, j])
        ax[1, j].plot(t, ev, "C1", lw=0.8, label="error")
        ax[1, j].fill_between(t, -sv, sv, color="C1", alpha=0.18, label=r"$\pm 2\sigma$")
        ax[1, j].set_title(f"joint velocity error, {JN[j]} (never measured)"); ax[1, j].set_ylabel("rad/s")
        ax[1, j].set_xlabel("t [s]")
    ax[0, 0].legend(); ax[1, 0].legend()
    fig.suptitle("Joint EKF: estimation error and reported uncertainty (nominal trial)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_estimation_timeseries.png")); plt.close(fig)


def fig_object(logs):
    fig, ax = plt.subplots(2, 3, figsize=(12, 5.4))
    for col, cond in enumerate(["nominal", "noise_x4", "cam_dropout"]):
        A = logs[cond]["log"]; t = A["t"]
        oe, osig, raw = A["obj_err"], A["obj_sig"], A["obj_raw_err"]
        for i, (lbl, sc) in enumerate((("x", 1e3), ("y", 1e3))):
            ax[0, col].plot(t, oe[:, i] * sc, f"C{i}", lw=0.9, label=f"KF {lbl}")
            ax[0, col].plot(t, raw[:, i] * sc, f"C{i}", lw=0.5, alpha=0.35, ls=":", label=f"raw {lbl}")
        s = 2 * osig[:, 0] * 1e3
        ax[0, col].fill_between(t, -s, s, color="k", alpha=0.10, label=r"$\pm 2\sigma$ (KF)")
        ax[0, col].set_title(f"object position error, {cond}"); ax[0, col].set_ylabel("mm")
        ax[1, col].plot(t, np.degrees(oe[:, 3]), "C3", lw=0.9, label="KF yaw")
        ax[1, col].plot(t, np.degrees(raw[:, 3]), "C3", lw=0.5, alpha=0.35, ls=":", label="raw yaw")
        s = 2 * np.degrees(osig[:, 3])
        ax[1, col].fill_between(t, -s, s, color="k", alpha=0.10)
        ax[1, col].set_title(f"object yaw error, {cond}"); ax[1, col].set_ylabel("deg"); ax[1, col].set_xlabel("t [s]")
    ax[0, 0].legend(ncol=2); ax[1, 0].legend()
    fig.suptitle("Object KF vs raw detection (dotted): filtering, uncertainty, and coasting when the camera is lost")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_object_kf.png")); plt.close(fig)


def fig_consistency(logs):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    A = logs["nominal"]["log"]; t = A["t"]
    n = 12
    lo, hi = chi2.ppf(0.025, n), chi2.ppf(0.975, n)
    ax[0].plot(t, A["nees_joint"], lw=0.6, color="C0")
    ax[0].axhline(n, color="k", ls="--", lw=1, label=f"E[NEES] = {n}")
    ax[0].axhspan(lo, hi, color="g", alpha=0.12, label="95 % band (single sample)")
    ax[0].set_title("joint EKF NEES"); ax[0].set_xlabel("t [s]"); ax[0].set_yscale("log"); ax[0].legend()
    lo6, hi6 = chi2.ppf(0.025, 6), chi2.ppf(0.975, 6)
    ax[1].plot(t, A["nis_joint"], lw=0.6, color="C1")
    ax[1].axhline(6, color="k", ls="--", lw=1, label="E[NIS] = 6")
    ax[1].axhspan(lo6, hi6, color="g", alpha=0.12)
    ax[1].set_title("joint EKF NIS (encoder innovation)"); ax[1].set_xlabel("t [s]"); ax[1].legend()
    m = np.isfinite(A["nis_obj"])
    ax[2].plot(t[m], A["nis_obj"][m], "o", ms=2.2, color="C2")
    ax[2].axhline(4, color="k", ls="--", lw=1, label="E[NIS] = 4")
    ax[2].axhline(18.47, color="r", ls=":", lw=1, label="gate (99.9 %)")
    ax[2].set_title("object KF NIS (camera innovation)"); ax[2].set_xlabel("t [s]"); ax[2].legend()
    fig.suptitle("Filter consistency on a nominal trial")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_consistency.png")); plt.close(fig)


def fig_tracking(logs):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 6))
    for name, c in (("nominal", "C0"), ("servo_only", "C3")):
        A = logs[name]["log"]; t = A["t"]
        err = np.sqrt(((A["q"] - A["q_ref"]) ** 2).sum(1))
        ax[0, 0].plot(t, err * 1e3, c, lw=0.9, label=f"{name}: RMS {1e3*np.sqrt((err**2).mean()):.2f} mrad")
        ce = np.linalg.norm(A["tcp"] - A["p_ref"][:, :3], axis=1)
        ax[0, 1].plot(t, ce * 1e3, c, lw=0.9, label=f"{name}: RMS {1e3*np.sqrt((ce**2).mean()):.2f} mm")
        ax[1, 0].plot(t, np.abs(A["tau"]).max(1), c, lw=0.9, label=name)
        ax[1, 1].plot(t[1:], np.abs(np.diff(A["u"], axis=0)).max(1) * 1e3, c, lw=0.7, label=name)
    ax[0, 0].set_title("joint tracking error |q - q_ref|"); ax[0, 0].set_ylabel("mrad"); ax[0, 0].legend()
    ax[0, 1].set_title("Cartesian tracking error |p_tcp - p_ref|"); ax[0, 1].set_ylabel("mm"); ax[0, 1].legend()
    ax[1, 0].set_title("peak joint torque"); ax[1, 0].set_ylabel("N m"); ax[1, 0].set_xlabel("t [s]"); ax[1, 0].legend()
    ax[1, 1].set_title("command increment |du| (chatter)"); ax[1, 1].set_ylabel("mrad/step"); ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].legend()
    fig.suptitle("Tracking with LQR on the estimated state vs the inner position servo alone")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_tracking.png")); plt.close(fig)


# ----------------------------------------------------------------- study figures
def fig_tuning():
    files = {k: os.path.join(RES, f"tuning_{k}.json") for k in ("ekf_q", "ekf_r", "obj_r", "lqr")}
    if not all(os.path.exists(f) for f in files.values()):
        print("tuning results missing, skipping tuning figures"); return
    T = {k: json.load(open(f)) for k, f in files.items()}
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for plant in sorted({r["plant"] for r in T["ekf_q"]}):
        rs = sorted([r for r in T["ekf_q"] if r["plant"] == plant], key=lambda r: r["accel_sigma"])
        a = [r["accel_sigma"] for r in rs]
        ax[0].plot(a, [r["nees_joint_mean"] for r in rs], "o-", ms=3, label=plant)
        ax[1].plot(a, [1e3 * r["qd_est_rmse"] for r in rs], "o-", ms=3, label=plant)
    ax[0].axhline(12, color="k", ls="--", lw=1, label="E[NEES] = 12"); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"process noise $\sigma_a$ [rad/s$^2$]"); ax[0].set_title("joint EKF consistency vs Q"); ax[0].legend()
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\sigma_a$ [rad/s$^2$]")
    ax[1].set_ylabel("mrad/s"); ax[1].set_title("velocity estimate RMSE vs Q"); ax[1].legend()
    rs = sorted(T["ekf_r"], key=lambda r: r["enc_r_scale"])
    s = [r["enc_r_scale"] for r in rs]
    ax[2].plot(s, [r["nees_joint_mean"] for r in rs], "o-", ms=3, label="NEES")
    ax[2].plot(s, [1e3 * r["qd_est_rmse"] for r in rs], "s-", ms=3, label="velocity RMSE [mrad/s]")
    ax[2].plot(s, [1e3 * r["u_chatter"] for r in rs], "^-", ms=3, label="command chatter [mrad]")
    ax[2].axhline(12, color="k", ls="--", lw=1)
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("assumed R / true R")
    ax[2].set_title("effect of mis-set encoder R"); ax[2].legend()
    fig.suptitle("Estimator tuning (each point = mean over randomised trials)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_tuning_estimator.png")); plt.close(fig)

    # object R + LQR trade-off
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    rs = sorted(T["obj_r"], key=lambda r: r["obj_r_scale"])
    s = [r["obj_r_scale"] for r in rs]
    ax[0].plot(s, [1e3 * r["obj_xy_err_at_grasp"] for r in rs], "o-", ms=3, label="object xy error at grasp [mm]")
    ax[0].plot(s, [r["obj_nees_mean"] for r in rs], "s-", ms=3, label="object NEES")
    ax[0].axhline(4, color="k", ls="--", lw=1, label="E[NEES] = 4")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("assumed R / fitted R")
    ax[0].set_title("object KF: mis-set camera R (4x noise plant)"); ax[0].legend()
    lq = [r for r in T["lqr"] if r.get("q_max")]
    base = [r for r in T["lqr"] if not r.get("q_max")]
    sc = ax[1].scatter([1e3 * r["tcp_track_rmse"] for r in lq], [r["effort_rms"] for r in lq],
                       c=[np.log10(r["q_max"] / r["u_max"]) for r in lq], cmap="viridis", s=34)
    for r in lq:
        ax[1].annotate(r["label"], (1e3 * r["tcp_track_rmse"], r["effort_rms"]), fontsize=5.5,
                       xytext=(2, 2), textcoords="offset points")
    for r in base:
        ax[1].scatter([1e3 * r["tcp_track_rmse"]], [r["effort_rms"]], marker="x", c="r", s=44, label="servo only")
    plt.colorbar(sc, ax=ax[1], label=r"$\log_{10}(q_{max}/u_{max})$")
    ax[1].set_xlabel("Cartesian tracking RMSE [mm]"); ax[1].set_ylabel("RMS joint torque [N m]")
    ax[1].set_title("LQR weight trade-off"); ax[1].legend()
    ax[2].scatter([1e3 * r["tcp_track_rmse"] for r in lq], [1e3 * r["u_chatter"] for r in lq], s=30, c="C0")
    for r in base:
        ax[2].scatter([1e3 * r["tcp_track_rmse"]], [1e3 * r["u_chatter"]], marker="x", c="r", s=44, label="servo only")
    ax[2].set_xlabel("Cartesian tracking RMSE [mm]"); ax[2].set_ylabel("command chatter [mrad/step]")
    ax[2].set_title("tracking vs command activity"); ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_tuning_control.png")); plt.close(fig)


def fig_moving(logs):
    """The moving-object result: velocity is estimated, never measured, and it is what makes the grasp work."""
    A = logs["moving"]["log"]; t = A["t"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 5.6))
    m = np.isfinite(A["obj_vel_true"][:, 0])
    for i, (lbl, c) in enumerate((("vx", "C0"), ("vy", "C1"))):
        ax[0, 0].plot(t[m], 1e3 * A["obj_vel_true"][m, i], c + "--", lw=1.4, label=f"true {lbl}")
        ax[0, 0].plot(t[m], 1e3 * A["obj_vel_hat"][m, i], c, lw=1.0, label=f"estimated {lbl}")
    ax[0, 0].set_ylabel("object velocity [mm/s]")
    ax[0, 0].set_title("object velocity: estimated vs true (never measured)"); ax[0, 0].legend(ncol=2)
    ax[0, 1].plot(t[m], 1e3 * np.linalg.norm(A["obj_vel_err"][m][:, :3], axis=1), "C3", lw=0.9)
    ax[0, 1].set_ylabel("|velocity error| [mm/s]"); ax[0, 1].set_ylim(0, 20)
    ax[0, 1].set_title("velocity estimate error (initial spike = filter starts at zero velocity)")
    for i, (lbl, c) in enumerate((("x", "C0"), ("y", "C1"))):
        ax[1, 0].plot(t, 1e3 * A["obj_err"][:, i], c, lw=0.9, label=f"KF {lbl}")
        ax[1, 0].plot(t, 1e3 * A["obj_raw_err"][:, i], c, lw=0.5, ls=":", alpha=0.4, label=f"raw {lbl}")
    s2 = 2e3 * A["obj_sig"][:, 0]
    ax[1, 0].fill_between(t, -s2, s2, color="k", alpha=0.10, label=r"$\pm2\sigma$")
    ax[1, 0].set_ylabel("object position error [mm]"); ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_ylim(-12, 12)
    ax[1, 0].set_title("position error, tracker vs raw detection"); ax[1, 0].legend(ncol=3, fontsize=6.5)
    ph = A["phase"]
    ax[1, 1].plot(t, ph, "C2", lw=1.2, drawstyle="steps-post")
    ax[1, 1].set_yticks(range(7)); ax[1, 1].set_yticklabels(["SCAN", "PREGRASP", "DESCEND", "GRASP", "LIFT", "DONE", "FAILED"], fontsize=6.5)
    ax[1, 1].set_xlabel("t [s]"); ax[1, 1].set_title("task phase")
    fig.suptitle("Tracking and grasping a moving object (0.04 m/s)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_moving_object.png")); plt.close(fig)


def fig_speed():
    fp = os.path.join(RES, "tuning_speed.json")
    rows = json.load(open(fp)) if os.path.exists(fp) else None
    if not rows:
        print("speed sweep missing, skipping"); return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for mdl, c, lbl in (("cv", "C0", "constant-velocity tracker"), ("static", "C3", "static-object filter (ablation)")):
        rs = sorted([r for r in rows if r["model"] == mdl], key=lambda r: r["speed"])
        sp = [1e3 * r["speed"] for r in rs]
        ax[0].plot(sp, [100 * r["success_rate"] for r in rs], "o-", color=c, ms=4, label=lbl)
        ax[1].plot(sp, [1e3 * r["obj_vel_rmse"] for r in rs], "o-", color=c, ms=4, label=lbl)
    ax[0].set_xlabel("object speed [mm/s]"); ax[0].set_ylabel("success rate [%]"); ax[0].set_ylim(-3, 103)
    ax[0].set_title("pick success vs object speed"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("object speed [mm/s]"); ax[1].set_ylabel("velocity RMSE [mm/s]")
    ax[1].set_title("object-velocity estimate error"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_speed_limit.png")); plt.close(fig)


def fig_robustness():
    files = sorted(glob.glob(os.path.join(RES, "mc_*.json")))
    if not files:
        print("no monte-carlo results"); return
    order = ["nominal", "noise_x2", "noise_x4", "noise_x8", "cam_dropout", "force_10N", "force_25N",
             "payload_0.5kg", "handeye_5mm_1deg", "handeye_10mm_2deg", "handeye_20mm_4deg",
             "abl_servo_only", "abl_no_objkf", "abl_no_objkf_noise_x4", "abl_p90_height_noise_x4",
             "abl_servo_only_force_25N"]
    data = {}
    for f in files:
        name = os.path.basename(f)[3:-5]
        data[name] = json.load(open(f))
    names = [n for n in order if n in data] + [n for n in sorted(data) if n not in order]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    xs = np.arange(len(names))
    rate, lo, hi = [], [], []
    for n in names:
        rs = data[n]; k = sum(r["success"] for r in rs)
        rate.append(100 * k / len(rs)); a, b = wilson(k, len(rs)); lo.append(100 * (k / len(rs) - a)); hi.append(100 * (b - k / len(rs)))
    cols = ["C7" if n.startswith("abl_") else "C0" for n in names]
    ax[0].bar(xs, rate, yerr=[lo, hi], capsize=2.5, color=cols)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names, rotation=38, ha="right", fontsize=6.6)
    ax[0].set_ylabel("success rate [%]"); ax[0].set_ylim(0, 105)
    ax[0].set_title("Success rate with 95 % Wilson intervals (grey = ablation)")
    bx = [1e3 * np.array([r.get("grasp_err_norm", np.nan) for r in data[n]], float) for n in names]
    bx = [b[np.isfinite(b)] for b in bx]
    ax[1].boxplot([b if len(b) else [np.nan] for b in bx], showfliers=True, widths=0.6,
                  flierprops=dict(ms=2, alpha=0.5))
    ax[1].set_xticks(xs + 1); ax[1].set_xticklabels(names, rotation=38, ha="right", fontsize=6.6)
    ax[1].set_ylabel("grasp alignment error [mm]"); ax[1].axhline(15, color="r", ls="--", lw=1, label="jaw capture limit")
    ax[1].set_title("Grasp error distribution"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_robustness.png")); plt.close(fig)


def main():
    todo = sys.argv[1:] or ["timeseries", "tuning", "robustness"]
    if "timeseries" in todo:
        logs = {}
        specs = dict(nominal=TrialConfig(seed=2, keep_log=True),
                     moving=TrialConfig(seed=2, speed=0.04, keep_log=True),
                     noise_x4=TrialConfig(seed=2, noise_scale=4.0, keep_log=True),
                     cam_dropout=TrialConfig(seed=2, dropout_after=2.5, keep_log=True),
                     servo_only=TrialConfig(seed=2, ctrl=ControllerConfig(use_lqr=False), keep_log=True))
        for k, tc in specs.items():
            logs[k] = run_trial(tc)
            print(f"  logged {k}: success={logs[k]['success']}")
        fig_estimation(logs); fig_object(logs); fig_consistency(logs); fig_tracking(logs); fig_moving(logs)
    if "tuning" in todo:
        fig_tuning(); fig_speed()
    if "robustness" in todo:
        fig_robustness()
    print("figures:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(RES, "fig_*.png"))))


if __name__ == "__main__":
    main()
