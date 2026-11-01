"""Generate docs/RESULTS.md: every number on the project page comes from results/*.json, not from prose.

Run after calibrate.py, characterise_noise.py, model_analysis.py, tuning.py and monte_carlo.py.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, CFG, DOCS = (os.path.join(ROOT, d) for d in ("results", "configs", "docs"))
JN = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def sec_model(o):
    m = load(os.path.join(RES, "model_analysis.json"))
    if not m:
        return
    o.append("## Part A - model\n")
    o.append("State `x = [q - q0, qdot]` (12), input `u` = joint position targets of the inner servo (6), "
             "output `y` = encoder angles (6).\n")
    o.append("| configuration | M diag [kg m^2] | pole range [1/s] | ctrb rank | min sv | obsv rank | min sv | obsv rank, 1 encoder |")
    o.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in ("home", "scan", "grasp"):
        r = m[k]
        o.append(f"| {k} | {np.round(r['M_diag'], 3).tolist()} | "
                 f"{min(r['poles_real']):.0f} .. {max(r['poles_real']):.0f} | "
                 f"{r['ctrb_rank']}/12 | {r['ctrb_sv_min']:.3f} | {r['obsv_rank']}/12 | {r['obsv_sv_min']:.2e} | "
                 f"{r['obsv_rank_one_encoder']}/12 |")
    v = m["validation"]
    o.append(f"\nServo gains: `kp = {m['servo']['kp']}`, `kd = {m['servo']['kd']}`, "
             f"torque limits `{m['servo']['force_limit']}` N m.\n")
    o.append(f"Linear model vs full MuJoCo, relative RMS error over a 0.6 s step response: "
             f"**{100*np.mean(v['step_0.02']['rel_rms_err_per_joint']):.2f} %** for a 0.02 rad step and "
             f"**{100*np.mean(v['step_0.3']['rel_rms_err_per_joint']):.1f} %** for a 0.3 rad step "
             f"(`results/model_validation.png`).\n")


def sec_sensing(o):
    c = load(os.path.join(CFG, "calibration.json"))
    n = load(os.path.join(CFG, "noise_model.json"))
    if not c:
        return
    o.append("## Part B - sensing\n")
    K = np.array(c["K"])
    o.append(f"Intrinsics from {c['n_views']} checkerboard views (7x6 inner corners, 25 mm squares), "
             f"pinhole with square pixels and the principal point fixed at the image centre:\n")
    o.append(f"* `fx = fy = {K[0,0]:.2f} px`, `cx = {K[0,2]:.1f}`, `cy = {K[1,2]:.1f}`")
    o.append(f"* reprojection RMS **{c['reproj_rms_px']:.3f} px**; focal-length error vs the simulator's true "
             f"value **{c['K_err_px']['fx']:+.2f} px**")
    o.append(f"* hand-eye `T_tool_cam` by Park's method: translation error **{c['handeye_trans_err_mm']:.2f} mm**, "
             f"rotation error **{c['handeye_rot_err_deg']:.2f} deg**\n")
    if n:
        o.append("Detection noise, split into a white part (varies between frames at a fixed pose) and a bias part "
                 "(varies with viewpoint, so averaging cannot remove it). Variances are fitted as `a + b r^2` "
                 "against camera-object range `r`:\n")
        o.append("| quantity | white `a` | white `b` | bias RMS |")
        o.append("| --- | --- | --- | --- |")
        o.append(f"| x, y | {n['xy_a']:.2e} m^2 | {n['xy_b']:.2e} | {1e3*np.sqrt(n['bias_xy']):.2f} mm |")
        o.append(f"| z | {n['z_a']:.2e} m^2 | {n['z_b']:.2e} | {1e3*np.sqrt(n['bias_z']):.2f} mm |")
        o.append(f"| yaw | {n['yaw_a']:.2e} rad^2 | {n['yaw_b']:.2e} | {np.degrees(np.sqrt(n['bias_yaw'])):.2f} deg |")
        o.append("\nThe bias term is added to the object KF's reported covariance; without it the filter is "
                 "over-confident, because a common-mode error passes straight through a static-state Kalman filter "
                 "(the gains sum to one).\n")


def _tun(name):
    return load(os.path.join(RES, f"tuning_{name}.json"))


def sec_moving(o):
    sp, oq = _tun("speed"), _tun("obj_q")
    if not sp and not oq:
        return
    o.append("## Moving object: tracking and interception\n")
    if sp:
        o.append("Pick success and object-velocity error versus object speed (constant-velocity tracker vs the "
                 "static-object filter). Velocity is never measured.\n")
        o.append("| object speed [mm/s] | success, CV tracker | success, static filter | velocity RMSE [mm/s] |")
        o.append("| --- | --- | --- | --- |")
        for v in sorted({r["speed"] for r in sp}):
            d = {r["model"]: r for r in sp if r["speed"] == v}
            o.append(f"| {1e3*v:.0f} | {100*d['cv']['success_rate']:.0f} % | {100*d['static']['success_rate']:.0f} % | "
                     f"{1e3*d['cv']['obj_vel_rmse']:.1f} |")
        o.append("")
    if oq:
        o.append("### Object-tracker process noise\n")
        scen = sorted({r["scenario"] for r in oq})
        o.append("| sigma_a [m/s^2] | " + " | ".join(scen) + " |")
        o.append("| --- " * (len(scen) + 1) + "|")
        for sa in sorted({r["obj_accel_sigma"] for r in oq}):
            row = {r["scenario"]: r for r in oq if r["obj_accel_sigma"] == sa}
            o.append(f"| {sa:g} | " + " | ".join(f"{100*row[c]['success_rate']:.0f} %" for c in scen) + " |")
        o.append("")


def sec_tuning(o):
    q, r, ob, lq = _tun("ekf_q"), _tun("ekf_r"), _tun("obj_r"), _tun("lqr")
    if not any((q, r, ob, lq)):
        return
    o.append("## Parts C/D - tuning studies\n")
    if q:
        o.append("### Joint EKF process noise `sigma_a`\n")
        o.append("| plant | sigma_a [rad/s^2] | NEES (E=12) | NIS (E=6) | qdot RMSE [mrad/s] | success |")
        o.append("| --- | --- | --- | --- | --- | --- |")
        for row in sorted(q, key=lambda x: (x["plant"], x["accel_sigma"])):
            o.append(f"| {row['plant']} | {row['accel_sigma']} | {row['nees_joint_mean']:.1f} | "
                     f"{row['nis_joint_mean']:.2f} | {1e3*row['qd_est_rmse']:.1f} | {100*row['success_rate']:.0f} % |")
        o.append("")
    if r:
        o.append("### Encoder R mis-set (the viva question: what if R were ten times smaller?)\n")
        o.append("| assumed R / true R | NEES | NIS | qdot RMSE [mrad/s] | chatter [mrad/step] | tracking RMSE [mm] | success |")
        o.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in sorted(r, key=lambda x: x["enc_r_scale"]):
            o.append(f"| {row['enc_r_scale']:g} | {row['nees_joint_mean']:.1f} | {row['nis_joint_mean']:.2f} | "
                     f"{1e3*row['qd_est_rmse']:.1f} | {1e3*row['u_chatter']:.2f} | "
                     f"{1e3*row['tcp_track_rmse']:.3f} | {100*row['success_rate']:.0f} % |")
        o.append("")
    if ob:
        o.append("### Object KF camera R mis-set (4x noise plant)\n")
        o.append("| assumed R / fitted R | object NEES (E=4) | xy error at grasp [mm] | success |")
        o.append("| --- | --- | --- | --- |")
        for row in sorted(ob, key=lambda x: x["obj_r_scale"]):
            o.append(f"| {row['obj_r_scale']:g} | {row['obj_nees_mean']:.2f} | "
                     f"{1e3*row['obj_xy_err_at_grasp']:.2f} | {100*row['success_rate']:.0f} % |")
        o.append("")
    if lq:
        o.append("### LQR weights (Bryson's rule: `Q_q = 1/q_max^2`, `R = 1/u_max^2`)\n")
        o.append("| weights | joint RMSE [mrad] | Cartesian RMSE [mm] | RMS torque [N m] | chatter [mrad/step] | success |")
        o.append("| --- | --- | --- | --- | --- | --- |")
        for row in lq:
            o.append(f"| {row['label']} | {1e3*row['track_rmse']:.3f} | {1e3*row['tcp_track_rmse']:.3f} | "
                     f"{row['effort_rms']:.2f} | {1e3*row['u_chatter']:.2f} | {100*row['success_rate']:.0f} % |")
        o.append("")


def sec_mc(o):
    files = sorted(glob.glob(os.path.join(RES, "mc_*.json")))
    if not files:
        return
    order = ["nominal", "noise_x2", "noise_x4", "noise_x8", "cam_dropout", "force_10N", "force_25N",
             "payload_0.5kg", "handeye_5mm_1deg", "handeye_10mm_2deg", "handeye_20mm_4deg",
             "moving_0.02", "moving_0.04", "moving_0.06", "oscillating", "moving_noise_x4",
             "moving_dropout", "moving_force_25N",
             "abl_servo_only", "abl_no_objkf", "abl_no_objkf_noise_x4", "abl_p90_height_noise_x4",
             "abl_servo_only_force_25N", "abl_mov_static_model", "abl_mov_no_objkf", "abl_mov_no_vel_ff",
             "abl_mov_no_lead"]
    data = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in files}
    names = [n for n in order if n in data] + [n for n in sorted(data) if n not in order]
    o.append("## Part E - robustness (Monte Carlo)\n")
    o.append("Each condition: independent trials with the cube pose drawn uniformly from "
             "x in [0.18, 0.27] m, y in [-0.20, 0.20] m, yaw in [-45, 45] deg.\n")
    o.append("| condition | n | success | 95 % CI | grasp err median [mm] | p95 [mm] | obj xy err at grasp [mm] | "
             "tracking RMSE [mm] | obj vel RMSE [mm/s] | mean time [s] |")
    o.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in names:
        rs = data[n]; k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        g = 1e3 * np.array([r.get("grasp_err_norm", np.nan) for r in rs], float)
        oe = 1e3 * np.array([r.get("obj_xy_err_at_grasp", np.nan) for r in rs], float)
        tr = 1e3 * np.array([r.get("tcp_track_rmse", np.nan) for r in rs], float)
        tt = np.array([r.get("time", np.nan) for r in rs], float)
        vr = 1e3 * np.array([r.get("obj_vel_rmse", np.nan) for r in rs], float)
        o.append(f"| {n} | {len(rs)} | {k}/{len(rs)} ({100*k/len(rs):.0f} %) | {lo:.0f}-{hi:.0f} % | "
                 f"{np.nanmedian(g):.2f} | {np.nanpercentile(g, 95):.2f} | {np.nanmedian(oe):.2f} | "
                 f"{np.nanmedian(tr):.3f} | {np.nanmedian(vr):.1f} | {np.nanmean(tt):.1f} |")
    o.append("")


def main():
    o = ["# Results", "",
         "Generated by `scripts/make_report.py` from the JSON files in `results/`. "
         "Regenerate everything with `make all` (see README).", ""]
    sec_model(o); sec_sensing(o); sec_tuning(o); sec_moving(o); sec_mc(o)
    out = os.path.join(DOCS, "RESULTS.md")
    open(out, "w").write("\n".join(o) + "\n")
    print("wrote", out, f"({len(o)} lines)")


if __name__ == "__main__":
    main()
