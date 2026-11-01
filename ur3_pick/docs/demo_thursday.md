# Thursday simulation demonstration — running order

Goal: pass the gate for the hardware induction, and show the EKF doing real work on a **moving** object,
which is what Teresa asked for. Budget about 10 minutes, keep each run under a minute.

Before you start: `export MUJOCO_GL=egl` is **not** wanted here — the live viewer needs a display, so just
run from a normal desktop terminal. Check it opens:

```bash
cd ur3_pick
../.venv/bin/python scripts/demo.py --speed 0.04        # viewer should open and the arm should pick
```

## Running order

| # | Command | What you say |
| --- | --- | --- |
| 1 | `scripts/demo.py` | Baseline: static cube, randomised pose. Five phases: SCAN, PREGRASP, DESCEND, GRASP, LIFT. The controller runs on the estimated state; `pytest` enforces that ground truth never reaches it. |
| 2 | `scripts/demo.py --speed 0.04` | **The main one.** Object moves at 40 mm/s. The tracker is a constant-velocity Kalman filter: position and yaw are measured at 10 Hz, **velocity is never measured** and is reconstructed. The arm leads the object on approach, then matches its velocity before closing. Point at the live velocity column. |
| 3 | `scripts/demo.py --speed 0.04 --static-model` | Same object, same everything, but the filter's process model says the object cannot move. **It fails** — 0/50 in the Monte Carlo, against 50/50 for the full pipeline. This is the evidence that the estimator is doing the work, not the controller. |
| 4 | `scripts/demo.py --speed 0.04 --dropout 3.0` | Camera is killed at 3 s. The constant-velocity model keeps propagating the object forward and the pick still completes; the covariance grows while it coasts. Then say the limitation out loud: this works because the velocity model is right, and it would drift if the object changed speed. |
| 5 | `scripts/demo.py --oscillate` | Object slides back and forth: the constant-velocity model is now *wrong* at every turning point, and you can see the estimate lag through the reversals. Honest about where the model breaks. |
| 6 | Show `results/fig_speed_limit.png` | The capability boundary: 100 % up to 40 mm/s, 93 % at 50, 27 % at 80, 0 % at 100. The static-model ablation is 0 % from 20 mm/s. Failures above 50 mm/s are the object leaving the reachable workspace before the arm can intercept, not tracking error — the velocity RMSE stays around 6 mm/s throughout. |
| 7 | Show `results/fig_robustness.png` | 50 trials per condition over 27 conditions, 1350 trials. Mention that hand-eye calibration error is the binding limit, which is the result that transfers directly to the real robot. |

## If asked "what is left to do"
Hardware: the controller only consumes an `Observation` (joint angles + RGB-D) and emits joint position
targets, so the simulator can be swapped for the real UR3e behind the same interface. The calibration and
noise-characterisation scripts are already written to run against whatever camera they are given.

## Fallbacks
* Viewer will not open → `scripts/demo.py --speed 0.04 --record /tmp/demo.mp4`, or play the clips already in
  `results/video/`.
* Machine is slow → drop `--realtime` (it is off by default, so runs go faster than wall clock).
* Anything crashes → `make test` still passes and `docs/RESULTS.md` has every number.
