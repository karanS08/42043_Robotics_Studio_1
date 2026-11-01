# Viva preparation

The marking guide lists the questions below. Every answer here is backed by a number in `docs/RESULTS.md`
or a figure in `results/`. Prepare to point at the figure, not just recite the number.

---

## Q1. "Point to a result on the project page and explain why it looks the way it does."

**Best answer: the moving-object ablation table (README §5).** On a static cube the object Kalman filter is
nearly redundant — 0.88 mm filtered against 0.89 mm raw, 50/50 either way. On a cube moving at 40 mm/s,
three separate ablations each give **0/50**: the static process model, no filter at all, and no velocity
feed-forward. The full pipeline gives 50/50.

*Why:* with a static target the filter is only averaging zero-mean noise, and the residual error is
dominated by calibration bias, which is common-mode and cannot be averaged away. With a moving target the
filter has to **predict**: the camera runs at 10 Hz and the controller at 100 Hz, so nine steps in ten are
driven by the process model, and the jaws only capture the object if the tool's speed matches to within
50 mm/s. A process model that says "the object cannot move" predicts a stationary object, the tool arrives
where the cube *was*, and the grasp never closes.

**Backup answer: the σ_a sweep (README §5).** Tuning the joint EKF's process noise on the nominal plant gives
NEES 12.7, almost exactly the expected 12 — it looks perfect. Add an unmodelled 0.5 kg payload and the same
filter reads NEES 3307.

*Why:* NEES measures estimation error against the covariance the filter reports. A payload makes the
prediction model wrong, so the error grows, but Q is not told about it, so the reported covariance does not.
The ratio blows up. Raising σ_a to 2 admits "my model may be wrong by about this much", which costs velocity
sharpness (1.1 → 4.5 mrad/s RMSE) and buys consistency across all three plants. I chose the robust point, not
the nominally optimal one.

## Q2. "Which states are observable from which sensors, and what happens when the marker leaves the view?"

* **Joint states.** Encoders measure `q` only, 6 of the 12 states. The pair `(A, C)` still has full rank 12
  at every configuration tested, because the servo stiffness couples position and velocity: `q̈` depends on
  `q` through `−M⁻¹(K_p + ∂g/∂q)`, so velocity shows up in the evolution of the measured positions.
  σ_min(obsv) ≈ 4×10⁻³, so it is observable but weakly — which is exactly why the velocity estimate is the
  part that degrades first when Q is mis-set.
* **With a single encoder** (shoulder pan only) the rank drops to **8–9 of 12**: the unmeasured joints are
  dynamically decoupled enough that their states cannot be reconstructed.
* **Object states.** Position and yaw are measured whenever the cube is inside the field of view and not
  clipped by the image border. The object's **velocity is never measured** — it is reconstructed from the
  sequence of noisy position measurements, exactly like the joint velocities. That reconstruction is what
  the whole moving-object result rests on.
* **When the camera is lost**, the filter stops correcting and *coasts* on its process model. For a moving
  object (`moving_dropout`, camera killed at 3 s) the velocity state carries the estimate forward and the
  pick still completes, 50/50 at 2.92 mm. For a static object (`cam_dropout`) it is 48/50 at 6.9 mm —
  slightly *worse* than the old static-object filter achieved, because a constant-velocity model integrates
  its own velocity noise while coasting. That is a real cost of using one tracker for both cases, and the
  σ_a table quantifies it: at σ_a = 0.8 the static-plus-dropout case drops to 20 %.
  Say the limitation plainly: coasting works because the constant-velocity model happens to be right, and
  an object that changed speed during the dropout would be tracked confidently and wrongly.

## Q3. "Where do the values in Q and R come from, and what would happen if R were ten times smaller?"

**R (encoders)** is the measured sensor noise, σ_enc = 1 mrad, squared — not a guess.
**R (camera)** is fitted, not assumed: 70 poses × 15 frames, variance regressed as `a + b·r²` on
camera-object range, then split into a white part and a viewpoint-bias part (README §4).
**Q** is a constant-acceleration model; σ_a is the one free parameter and was chosen by the sweep in Q1.

**If R were ten times smaller** (`enc_r_scale = 0.1`, measured, not predicted):

| | R×1 | R×0.1 | R×0.01 |
| --- | --- | --- | --- |
| NEES (E = 12) | 2.4 | 47.5 | 670 |
| NIS (E = 6) | 5.4 | 44.0 | 308 |
| velocity RMSE [mrad/s] | 4.5 | 21.2 | 84.9 |

The filter believes the encoders more than it should, so the gain rises, measurement noise is differentiated
straight into the velocity estimate (4.7× worse), and both consistency statistics blow up. Tracking barely
changes because the LQR is dominated by the position states, which are measured directly — the damage is
confined to the unmeasured states. Going the other way (R×10) *lowers* velocity RMSE to 1.5 mrad/s because
the true trajectory is smooth, but NEES falls to 0.17: the filter becomes badly over-confident in the wrong
direction and would ignore a real fault.

## Q4. "How were the controller gains or LQR weights chosen, and what limits them?"

Bryson's rule: `Q_q = 1/q_max²` with q_max = 5 mrad of tolerable joint error, `R = 1/u_max²` with
u_max = 20 mrad of tolerable command. Then verified by sweep (README §6): LQR gives 0.70 mm Cartesian RMSE
versus 1.18 mm for the inner servo alone, at the same torque.

**What limits them:** pushing to q_max = 1 mrad / u_max = 20 mrad reaches 0.57 mm but **diverged in 1 of 15
trials** (NaN). Three things bound the gain: the ±56/28/12 N·m torque limits (which are outside the linear
model the gain is designed from), the 100 Hz sample rate, and the fact that the feedback acts on an
*estimate* — high gain amplifies estimation error, so the estimator's quality caps the usable controller
bandwidth. Defect #1 in README §7 is exactly this failure mode in its extreme form.

## Q5. "Walk through the frames from the sensor measurement to the quantity the controller acts on."

Use `results/fig_frames.png`, left to right:

1. **Pixel** `(u, v)` plus depth `d` from the 320×240 RGB-D image, 10 Hz.
2. **Camera frame** `p_cam = K⁻¹[u v 1]ᵀ d`, with `K` from checkerboard calibration (0.19 px RMS). OpenCV
   convention: x right, y down, z forward.
3. **Tool frame** via the hand-eye transform `T_tool_cam`, from `cv2.calibrateHandEye` (1.03 mm, 0.16°).
   This is calibrated offline and is the dominant *systematic* error in the chain.
4. **Base frame** via `T_base_tool(q̂)` — forward kinematics evaluated at the **EKF estimate**, never at the
   true joint angles. Encoder noise therefore enters the object estimate through the kinematics.
5. **Object filter** fuses that measurement into `[x, y, z, ψ]` with a gated, range-dependent R.
6. **Goal** `p_ref`: phase logic picks pre-grasp / descend / lift, a second-order filter smooths it.
7. **Joint reference** `q_ref` by damped-least-squares IK.
8. **Command** `u = u_ff(q_ref) − K(x̂ − x_ref)`, a position target for the inner servo.

Note the two places the estimate enters: the object measurement is expressed through `FK(q̂)`, and the
feedback term uses `x̂`. Neither touches ground truth.

## Q6. "What failed during development, how was it diagnosed, and what would you change?"

Four defects, README §7. The strongest one to tell:

**The LQR diverged in 7 control steps.** Diagnosis: the commands railed immediately, so I compared the
finite-difference linearisation against an analytic `A`, `B` built from `M(q)`, `K_p`, `K_d` — they disagreed
by 0.16. That pointed at the plant, not the gain: the stock servo damping saturates the joint torque limits
above ≈0.28 rad/s, so the model the Riccati equation was solved from did not describe the real plant at all.
Fix: retune `k_d` to critical damping `2√(k_p M_ii)` and compute the gain from the analytic model.

**What I would change.** (i) Put the hand-eye transform into the estimator as an online-calibrated state —
it is the binding accuracy limit and is currently fixed. (ii) Model finger contact properly instead of a
weld. (iii) Give the object filter velocity states so it degrades gracefully when the object is not static.

---

## Likely follow-ups

* *"Show me where ground truth is used."* `eval/runner.py` only. `tests/test_firewall.py` enforces it with an
  AST check that `core/` never imports the plant and never touches simulator state it does not own, plus a
  test that drives the controller with a duck-typed observation object that has no access to the simulator.
* *"Why PBVS and not IBVS?"* PBVS keeps the task in Cartesian space, so the reference filter, the IK and the
  LQR all operate on quantities with physical units and the frame chain is explicit. IBVS would avoid the
  depth back-projection but couple the controller to image-plane dynamics and the interaction matrix; given
  one person and the schedule, PBVS was the defensible choice. The comparison is listed as future work.
* *"Why does the lead term not matter?"* It was my own assumption that aiming at `p̂ + v̂·t_lead` on approach
  would be necessary; the ablation says otherwise — removing it is still 50/50, because the velocity
  feed-forward already drives the tool onto the object's trajectory before the descent starts. Leaving the
  lead on *during* the descent is actively harmful: it placed the jaws `t_lead × speed` ahead of the cube
  and cost 11.6 mm of alignment error until it was fixed. Reporting the ablation that contradicted me is
  the point.
* *"Is 50 trials enough?"* Wilson intervals are reported for every condition; at 50/50 the 95 % interval is
  93–100 %, so the claim is "≥93 %", not "100 %".
* *"Why is the object filter NEES below its expected value?"* It is conservative by design — the viewpoint
  bias is added as if fully present every trial, while part of it averages out as the camera moves. Safe for
  gating, but it does mean the reported uncertainty is pessimistic. Stated in README §5.
