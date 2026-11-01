# Hardware bring-up — UR3e

**Status: written against the real interfaces, but never executed on a robot.** Every stage below is
designed to fail safely and to be verified before the next one. Do not skip a stage.

> **Safety.** The robot moves under a controller driven by a *vision estimate*. Keep the e-stop in hand,
> keep the workspace clear, and run each stage once at reduced speed before trusting it. The software
> envelope in `configs/hardware.json` (`ws_*`, `max_joint_speed`, `max_step`, `max_tracking_error`) is a
> backstop, not a substitute for the robot's own safety configuration. Set conservative joint and TCP
> limits on the teach pendant as well.

## 0. One-time setup

```bash
.venv/bin/pip install -r requirements-hw.txt      # ur_rtde + pyrealsense2
```

On the teach pendant: power on, release brakes, and set the robot to **Remote Control** mode. Note the IP
under *Settings → System → Network*. The PC must be on the same subnet.

`configs/hardware.json` already has `"camera": "realsense"`, which is the camera in the lab, and
`"gripper": "none"`. **The gripper does not need to be known in advance** — stage 1 probes for it (below).
So the only thing to supply is the IP.

If the wrist camera is ever not a RealSense, implement `read()` in a new class in `hardware/devices.py`
returning `(rgb uint8 HxWx3, depth float32 HxW metres)` — that is the entire camera contract.

## 1. Connect and verify — **no motion**

```bash
make hw-check IP=192.168.1.102
```

Checks ping, RTDE ports, robot/safety mode, joint reads, FK against the robot's own TCP, the gripper, the
camera (frame rate, depth coverage) and the calibration files. It writes `configs/hardware.json`, so later
commands only need `IP=` if it changes.

### The gripper is detected, not assumed

`detect_gripper()` runs two probes and prints both:

* **Robotiq** — opens the URCap socket (port 63352) and sends `GET STA` / `GET POS`. A reply of the form
  `STA <n>` is conclusive evidence of a Robotiq, and `n = 3` means it is already activated. These probes are
  **read-only on purpose**: the code never sends `SET ACT 1` during a probe, because activating a Robotiq
  makes the jaws run a full open/close calibration stroke, and a stage-1 check must not move anything.
* **Digital output** — reports whether the RTDE IO interface is open and what the output bits read. This is
  *never* chosen automatically: every UR has digital outputs, so their presence is no evidence a gripper is
  wired to one.

If a Robotiq is found, `--save` writes `"gripper": "robotiq"` into `configs/hardware.json` and every later
stage uses it. If nothing is found and a gripper is nonetheless fitted, it is probably on a digital output —
set `"gripper": "digital_out"` and `gripper_do` to its index, then re-run. An unknown key in that file is
reported as a likely typo and ignored rather than raising `TypeError` in front of the robot.

Only after a gripper has been identified does the check actually `connect()` it, which for a Robotiq does
perform the one activation stroke. That is the first motion of any kind in the bring-up.

Expected snags:
* *FK model vs robot TCP differs by ~150 mm* — normal until a tool centre point is configured on the
  pendant. The project's TCP is 150 mm along the tool z from the flange (`sim/scene.py: TCP_Z`). Either set
  that TCP on the pendant or keep the difference in mind; the controller uses its own model either way.
* *Depth coverage below 50 %* — the RealSense minimum range is about 0.2 m; the pre-grasp hover is 0.10 m
  above the object, so expect invalid depth close in. Raise `pregrasp_height` if the object disappears.

## 2. Identify the inner servo loop — **the robot moves, small steps**

```bash
make hw-identify IP=192.168.1.102
```

Commands 0.03 rad steps per joint through `servoJ` and fits `M q̈ = k_p(u − q) − k_d q̇`, writing
`servo_kp` / `servo_kd` into `configs/hardware.json` and a figure to `results/hw_servo_identification.png`.

This is what makes Part A's model a model of **this robot** rather than of MuJoCo. Until it is run, the
controller falls back to `feedforward="none"`, which simply passes the joint reference to `servoJ` and keeps
only the LQR correction — safe, and measured at 6/6 success in simulation, but with roughly double the
grasp error (2.19 mm against 1.07 mm).

## 3. Calibrate the real camera — **no motion in manual mode**

```bash
make hw-calib IP=192.168.1.102          # manual: freedrive to each pose, press ENTER
# or
.venv/bin/python scripts/hw_calibrate.py --ip ... --auto --views 25
```

Place a 7×6-inner-corner checkerboard flat in the workspace. Collect at least 15 views covering a range of
tilts and distances. Targets: **reprojection RMS < 0.5 px** and **hand-eye spread < 5 mm**.

That spread is the number that matters most. In simulation, grasp error tracks hand-eye error almost
one-for-one and the task collapses once the error passes the jaw envelope (20 mm → 20 % success). If the
spread is above ~10 mm, recalibrate before running the pick; it will not work.

The simulation calibration is backed up to `configs/calibration.sim_backup.json`, and the new file is
tagged `source: "hardware"`. `hw_run.py` warns loudly if it sees a simulation calibration.

## 3b. Fit the colour thresholds to the lab lighting — **no motion**

```bash
.venv/bin/python scripts/hw_tune_segment.py --ip 192.168.1.102          # look first
make hw-segment IP=192.168.1.102                                        # then write configs/segment.json
```

The detector calls a pixel "cube" when it is bright red and much redder than green or blue. The three
thresholds that do that in the simulator do not survive a change of lighting: fluorescent tubes desaturate
the cube, a window behind it clips the red channel, and auto-exposure shifts the whole histogram between
frames. This is the single most likely thing to stop a hardware run, so it is fitted from images rather
than guessed at the bench.

The script grabs frames from the wrist camera, finds the cube by **hue and saturation** — independent of
the thresholds being fitted, since a red hue stays red whatever the exposure — and reads the thresholds off
that blob's own statistics. It then checks each candidate the way the detector will use it and reports:

* `detected` — in how many frames the fitted mask still finds the cube (want all of them),
* `iou` — agreement between the fitted mask and the hue blob (want > 0.5, ideally > 0.9),
* `leak_px` — background pixels wrongly called cube (want near zero).

It prints the mask as ASCII, so it is readable over SSH with no display, and saves
`results/segment/frame*_mask.png` with the mask in green. Nothing is written without `--write`.

If no red blob is found by hue either, the cube is not in view, the lighting is too warm, or the camera is
returning BGR where RGB is expected — the script says so and saves the frame to look at.

Worth knowing: run on **simulator** frames (`--sim`, which also self-tests the script), the fit raises IoU
from 0.80 to 0.98 — the hardcoded defaults clip the dimmer edge pixels of the cube's top face.
`configs/segment.json` is deliberately left absent in the repo so the published simulation results stay
reproducible against the defaults. Writing it changes the detector, so if you ever write it for
simulation, regenerate the results with `make all`.

```bash
make hw-noise IP=192.168.1.102          # then measure this camera's detection noise -> object filter R
```

## 4. Dry run — **no motion, full pipeline**

```bash
make hw-dry IP=192.168.1.102
```

Opens the camera and the *receive* side of RTDE only; the control interface is never created, so nothing can
move. It runs the detector, both filters and the controller, and prints the commands it would send. Check:

* the object is detected from the scan pose (not `(no detection)`),
* the estimated position is physically sensible in the base frame,
* for a moving object, the estimated speed matches reality,
* `|u − q|` stays small — a large residual means the controller is fighting the model.

If detection fails, go back to stage 3b — the colour thresholds are the most likely cause, and
`hw_tune_segment.py` fits them from the camera's own frames. There is no longer any reason to edit
`core/sensing.py` by hand for this.

## 5. Run it — **the robot moves**

```bash
make hw-run IP=192.168.1.102            # prompts for confirmation
```

Starts with a blocking `moveJ` to the home pose, then servos at 100 Hz. Defaults are deliberately slower
than simulation (`--v-max 0.08` against 0.25 m/s). Every command passes the safety limiter; any violation
calls `servoStop` and aborts. Each run is logged to `results/hw_<timestamp>.npz`.

Suggested order: static object first, then a slowly moving one.

## 6. Compare against simulation

```bash
make hw-compare LOG=results/hw_20261022_143500.npz
```

Produces `results/fig_sim_vs_hardware.png` and a table of the quantities measurable on both — joint
tracking, command activity, reported object σ, phase timings, cycle time. Grasp alignment error and NEES
need ground truth, so they are reported for simulation only and left blank for hardware. The marking guide
asks for exactly this comparison.

## What is genuinely untested

Every line in `hardware/` has been syntax-checked, the dependencies are installed and import cleanly
(`ur_rtde` 1.6.0, `pyrealsense2` 2.56.5), and the interface is covered by
`tests/test_firewall.py::test_hardware_backend_matches_the_simulator_interface` — but none of it has talked
to a robot. What remains untested, in the order it is likely to bite:

| Untested | Why it is a risk | What it needs |
|---|---|---|
| `servoJ` timing loop | Holding 100 Hz depends on the real network and the controller's own cycle | Stage 4 dry run; watch for late frames |
| Hand-eye transform | Physical camera mount is unknown; sim grasp error tracks this error one-for-one | Stage 3, spread < 5 mm |
| Colour thresholds | Lab lighting, not sim lighting | Stage 3b, now fitted automatically |
| Camera model | **Confirmed Intel RealSense**, already the configured default | nothing to do |
| Gripper handshake | Model still unconfirmed, so stage 1 probes for it; a non-Robotiq on a digital output needs `gripper_do` set by hand | Stage 1, then stage 5 |
| RealSense minimum range | ~0.2 m typical; the pre-grasp hover is 0.10 m, so depth may drop out during DESCEND | Stage 4; the object filter should coast through it — verify, do not assume |
| Identified servo gains | `servo_kp/kd` default to the simulator's retuned values | Stage 2 |

Budget a full session for stages 1–3b before expecting a pick.
