# 41014 Sensors and Control — Assessment 4: Integrated Project

**Track B — vision-guided pick with a UR3e**

| | |
| --- | --- |
| Student | Karan Sharma |
| Student ID | 26276485 |
| Group | 35 |
| Team members | 1 — Karan Sharma (26276485), working solo as a master's student enrolled in 42043 |
| Subject | 41014 Sensors and Control in Mechatronics Systems |
| Platform | UR3e + eye-in-hand RGB-D camera, MuJoCo 3.15, with a real-hardware backend over RTDE |

An eye-in-hand RGB-D camera on a UR3e. The robot searches the table, localises a randomly placed cube —
**static or moving** — and picks it up.

**Every control decision is taken on estimated state.** Joint velocity is never measured. The object pose
comes from a Kalman filter fed by the camera. The simulator's ground truth is used only for evaluation,
and that boundary is enforced mechanically by an AST analysis in `ur3_pick/tests/test_firewall.py` rather
than by inspection.

**Headline result.** On a *static* cube the object filter is nearly redundant. On a cube moving at 40 mm/s,
removing it — or replacing it with a static-object model, or dropping the velocity feed-forward — takes the
pick from **50/50 to 0/50**. Velocity is never measured; it is reconstructed.

![pipeline](ur3_pick/results/fig_frames.png)

---

## Watch it run

Left half of each clip is the scene; right half is what the estimator sees — wrist camera, object estimate
against truth with its ±2σ band, and the shoulder-lift velocity that is **never measured**.

### The headline: the same moving cube, with and without a velocity model

| Constant-velocity object filter — **50/50** | Static-object model instead — **0/50** |
|---|---|
| ![moving object, CV filter](ur3_pick/results/gif/moving.gif) | ![moving object, static model](ur3_pick/results/gif/moving_static_model.gif) |
| The filter reconstructs the cube's velocity from position measurements alone and the controller leads the target. | Identical code, one changed process model. The estimate lags behind the cube and the gripper closes on empty table. |

### Nominal pick, static cube

![nominal pick](ur3_pick/results/gif/nominal.gif)

### Robustness

| Condition | |
|---|---|
| **Camera dropout** — frames cut for 1 s mid-approach; the filter coasts on its own prediction | ![camera dropout](ur3_pick/results/gif/cam_dropout.gif) |
| **4× detector noise** | ![4x noise](ur3_pick/results/gif/noise_x4.gif) |
| **25 N external force** applied to the arm | ![25 N disturbance](ur3_pick/results/gif/force_25N.gif) |
| **Oscillating object** | ![oscillating](ur3_pick/results/gif/oscillating.gif) |
| **Moving object + camera dropout** together | ![moving with dropout](ur3_pick/results/gif/moving_dropout.gif) |

<sub>These are GIFs because GitHub will not play an mp4 that lives in a repository — only files uploaded
through its web UI get a player. The original H.264 renders are in
[`ur3_pick/results/video/`](ur3_pick/results/video/) at full 1240×480; regenerate the GIFs with
`make gifs`.</sub>

---

## Repository layout

```
ur3_pick/            the project
├── README.md        FULL technical report — model, sensing, estimation, control, results
├── sim/             MJCF scene assembly, plant, sensors, disturbances, ground truth
├── core/            kinematics, sensing/calibration, EKF + object KF, controller
├── eval/            trial runner — the only place estimates meet ground truth
├── hardware/        real UR3e backend: RTDE, RealSense, gripper, safety limiter
├── scripts/         experiments (sim) and the hw_* bring-up sequence (real robot)
├── tests/           ground-truth + backend-agnosticism firewall (7 tests)
├── configs/         calibration, noise model, segmentation, hardware settings
├── docs/            hardware bring-up, viva prep, video script, AI declaration
└── results/         every figure, Monte Carlo log and video in the report
```

Start with **[`ur3_pick/README.md`](ur3_pick/README.md)** — that is the real report. This page is the front
door and the command reference.

---

## Setup

```bash
git clone https://github.com/karanS08/42043_Robotics_Studio_1.git
cd 42043_Robotics_Studio_1

python3 -m venv .venv
.venv/bin/pip install -r ur3_pick/requirements.txt
export MUJOCO_GL=egl            # or osmesa for software rendering
```

### Third-party model

The UR3e description comes from
[SouthColumn76/universal_robots_ur3e](https://github.com/SouthColumn76/universal_robots_ur3e), originally
ROS Industrial Consortium, **BSD-3-Clause**. The MJCF and its 20 collision/visual meshes are vendored into
`ur3_pick/model/` so this repository is self-contained — clone it and it runs, with no second clone and no
dangling symlink. The licence is redistributed verbatim as
[`ur3_pick/model/LICENSE_ur3e.txt`](ur3_pick/model/LICENSE_ur3e.txt), as clause 1 requires.

`ur3_pick/model/ur3e.xml` carries exactly **two documented edits**, both recorded in a header comment in
the file itself:

1. the keyframe was removed, so the scene can add free bodies;
2. the servo damping `biasprm[2]` was retuned `200/100/40 → 42/24/9` to critical damping — the stock values
   saturated the torque limits above ~0.28 rad/s and invalidated the linearisation the LQR is built on.

---

## Simulation

All commands run from `ur3_pick/`.

```bash
cd ur3_pick

.venv/bin/python scripts/run_one.py 0        # one pick, prints every metric
.venv/bin/python scripts/demo.py             # live MuJoCo viewer (needs a display)
make test                                    # the 7 firewall tests
make all                                     # regenerate every number and figure in the report
```

`make all` runs calibration → noise characterisation → model analysis → tuning → 1350 Monte Carlo trials →
plots → report. About 9 minutes on 8 workers.

**Memory cap.** MuJoCo is heavy and long runs previously OOM-killed this laptop. Every heavy job goes
through `scripts/capped.sh`, which runs it inside a `systemd-run` scope with a hard `MemoryMax` and swap
disabled:

```bash
bash scripts/capped.sh --mem 4G .venv/bin/python scripts/monte_carlo.py 50
```

The Makefile already applies a 5 GB cap at 8 workers. RSS stays flat at ~466 MB per worker.

---

## Hardware — running on the real UR3e

Full detail, including what each check means and what to do when it fails:
**[`ur3_pick/docs/hardware_bringup.md`](ur3_pick/docs/hardware_bringup.md)**.

### Before you start

On the teach pendant: power on, release the brakes, set the robot to **Remote Control** mode. Read the IP
from *Settings → System → Network*. The PC must be on the same subnet.

`configs/hardware.json` already has `"camera": "realsense"`. **The gripper does not need to be known in
advance** — stage 1 probes for it. The only thing to supply is the IP.

```bash
cd ur3_pick
.venv/bin/pip install -r requirements-hw.txt     # ur_rtde + pyrealsense2 (prebuilt wheels, no compile)
export IP=192.168.1.102                          # <- your robot's address
```

### The sequence

Run these **in order**. The "moves" column is the one to read before pressing enter.

| # | Command | Moves? | What it does |
|---|---|---|---|
| 1 | `make hw-check IP=$IP` | **no** | Ping, RTDE ports, robot/safety mode, joint reads, FK vs the robot's own TCP, **gripper probe**, camera frame rate and depth coverage, calibration provenance. Writes `configs/hardware.json`. |
| 2 | `make hw-identify IP=$IP` | **yes** — small steps | Identifies the inner position-servo gains `k_p`, `k_d` from step responses, so the controller's feed-forward matches this robot instead of the simulator. |
| 3 | `make hw-calib IP=$IP` | no (freedrive by hand) | Intrinsics + hand-eye from a 7×6-inner-corner checkerboard. **Targets: reprojection RMS < 0.5 px, hand-eye spread < 5 mm.** |
| 3b | `make hw-segment IP=$IP` | **no** | Fits the cube's colour thresholds to the lab lighting from the camera's own frames. |
| 3c | `make hw-noise IP=$IP` | **yes** — moves between poses | Measures this camera's real detection noise, fits `σ² = a + b·range²`, writes the object filter's `R`. |
| 4 | `make hw-dry IP=$IP` | **no** | **The whole pipeline, no motion.** Camera, detector, both filters, controller — prints the commands it *would* send. The control interface is never opened. |
| 5 | `make hw-run IP=$IP` | **yes** — the pick | Blocking `moveJ` home, then 100 Hz servo. Prompts for confirmation. Logs to `results/hw_<timestamp>.npz`. |
| 6 | `make hw-compare LOG=results/hw_<timestamp>.npz` | no | Sim vs hardware figure and table. |

Budget a full session for stages 1–3c before expecting a pick.

### Reading stage 4 before you let it move

The dry run is the gate. Check all four:

- the object is detected from the scan pose — not `(no detection)`,
- the estimated position is physically sensible in the base frame,
- for a moving object, the estimated speed matches reality,
- `|u − q|` stays small — a large residual means the controller is fighting the model.

### Safety

Every command passes `hardware/safety.py` before reaching the robot: joint limits, per-step and per-second
speed caps, a TCP bounding box, and a maximum tracking error. Any violation calls `servoStop` and raises
`SafetyAbort`. Hardware defaults are deliberately slower than simulation — `--v-max 0.08` against 0.25 m/s.

`HardwarePlant` has **no `.truth` attribute**, so any code that reaches for ground truth on real hardware
fails loudly instead of silently reading nothing.

### Honest status

The hardware backend is written, its dependencies install and import, and it is interface-checked against
the simulator by the test suite — but **it has not yet been validated against a robot**. What is still
untested, and the stage that closes each:

| Untested | Risk | Closed by |
|---|---|---|
| `servoJ` 100 Hz timing loop | Real network and controller cycle | Stage 4 |
| Hand-eye transform | Physical mount unknown; grasp error tracks this error 1:1 | Stage 3 |
| Colour thresholds | Lab lighting ≠ sim lighting | Stage 3b |
| Gripper handshake | Model unconfirmed; stage 1 probes for it | Stages 1, 5 |
| RealSense minimum range | ~0.2 m typical vs a 0.10 m pre-grasp hover | Stage 4 |
| Servo gains | Default to the simulator's values | Stage 2 |

---

## Documentation

| Document | What it covers |
|---|---|
| [`ur3_pick/README.md`](ur3_pick/README.md) | The full report: model, sensing, estimation, control, failure analysis, robustness |
| [`ur3_pick/docs/RESULTS.md`](ur3_pick/docs/RESULTS.md) | Generated results tables |
| [`ur3_pick/docs/hardware_bringup.md`](ur3_pick/docs/hardware_bringup.md) | Stage-by-stage hardware procedure |
| [`ur3_pick/docs/video_script.md`](ur3_pick/docs/video_script.md) | 5-minute video script |
| [`ur3_pick/docs/AI_declaration.md`](ur3_pick/docs/AI_declaration.md) | Generative-AI use declaration |
| [`ur3_pick/docs/acknowledgements.md`](ur3_pick/docs/acknowledgements.md) | Third-party material |

## Contribution statement

Solo project. All modelling, implementation, experiments, analysis and documentation by the author.
Third-party material is listed in [`ur3_pick/docs/acknowledgements.md`](ur3_pick/docs/acknowledgements.md);
generative-AI use is declared in [`ur3_pick/docs/AI_declaration.md`](ur3_pick/docs/AI_declaration.md).
