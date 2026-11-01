# 5-minute demo video — shot list and narration

Clips are in `results/video/` (1240×480, 25 fps). Each shows, side by side: the external view, the wrist
camera the estimator actually sees, the object estimate against truth with ±2σ, and — for the moving-object
clips — the estimated object velocity against the truth, a state that is never measured. Record the
voice-over over these. **Content beyond 5 minutes is not considered**, so keep to the timings.

| # | Time | Clip | What to say |
| --- | --- | --- | --- |
| 1 | 0:00–0:25 | `fig_frames.png` (still) | Track B, UR3e, eye-in-hand RGB-D. Task: find a randomly placed cube, static or moving, and pick it. State the rule once: the controller never sees ground truth, and `tests/test_firewall.py` enforces it mechanically. Walk the frame chain left to right in one sentence. |
| 2 | 0:25–1:10 | `nominal.mp4` | Static baseline. Narrate the phases as the badge changes: SCAN → PREGRASP → DESCEND → GRASP → LIFT. The object trace is the estimate against truth, inside ±2σ. The bottom trace is shoulder-lift **velocity error** — a state the encoders never measure. Grasp error 1.07 mm, 50/50 trials. |
| 3 | 1:10–2:10 | `moving.mp4` | **The core estimation result.** Object moves at 40 mm/s. Position and yaw are measured at 10 Hz; velocity is never measured and is reconstructed by a constant-velocity Kalman filter. Nine control steps in ten run on prediction alone. Point at the velocity trace locking on within ~0.1 s, then at the tool matching the object's speed before the jaws close. 50/50 trials. |
| 4 | 2:10–2:40 | `moving_static_model.mp4` | Same scene, same controller, one change: the filter's process model says the object cannot move. **It fails.** Then give the three ablations together — static model, no filter, no velocity feed-forward — each **0/50**, against 50/50 for the full pipeline. This is the evidence that the estimator is load-bearing, not decorative. |
| 5 | 2:40–3:05 | `moving_dropout.mp4` | Camera killed at 3 s. The velocity state carries the estimate forward and the pick still completes (50/50, 2.92 mm). Say the limitation out loud: this works because the constant-velocity model happens to be right, and it would drift if the object changed speed. |
| 6 | 3:05–3:25 | `oscillating.mp4` | Object slides back and forth, so the constant-velocity assumption is wrong at every turning point and you can see the estimate lag through the reversals. It still succeeds (50/50) because the reversals are away from the grasp. Honest about where the model breaks. |
| 7 | 3:25–4:05 | `fig_speed_limit.png` + `fig_robustness.png` (stills) | Capability boundary: 100 % to 40 mm/s, 93 % at 50, 27 % at 80, 0 % at 100; the static-model ablation is 0 % from 20 mm/s. Stress that the failures above 50 mm/s are **not** tracking failures — velocity RMSE stays ~6 mm/s — the object simply leaves the reachable workspace first. Then the robustness chart: hand-eye calibration error is the binding limit, collapsing to 20 % at 20 mm. |
| 8 | 4:05–4:40 | `fig_tuning_estimator.png` or the σ_a table (still) | One quantified trade-off. Either the joint-EKF σ_a (nominal-optimal tuning is 2 orders of magnitude inconsistent under a payload) or the object-tracker σ_a (0.8 → 20 % on static-plus-dropout, 0.02 → 93 % on oscillating; 0.05 is the knee). |
| 9 | 4:40–5:00 | `fig_consistency.png` (still) | One failure found and fixed — the 90th-percentile height estimator, 6.42 mm against 1.59 mm at ×4 noise. Close with the limits: welded grasp, kinematically driven object, constant-velocity assumption, no hardware yet. |

## Producing the file

```bash
make video                      # regenerates all eight clips (~3 min)
```

Assemble in any editor (Kdenlive, Shotcut, DaVinci): drop the clips in the order above, insert the stills
with 15–30 s holds, record the voice-over in one take against the timeline, export 1080p H.264.
