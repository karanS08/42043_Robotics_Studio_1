"""Stage 1 of hardware bring-up: connect and verify. THIS SCRIPT NEVER MOVES THE ROBOT.

    python scripts/hw_check.py --ip 192.168.1.102

Runs through, in order: network reachability, RTDE receive, robot/safety mode, RTDE control, joint and TCP
readings cross-checked against the kinematic model, gripper, camera, and the calibration files. Prints a
PASS/FAIL table and writes configs/hardware.json so later scripts just need --ip once.
"""
import argparse, os, socket, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from hardware.config import HardwareConfig

OK, BAD, WARN = "  PASS", "  FAIL", "  WARN"
ROBOT_MODES = {0: "NO_CONTROLLER", 1: "DISCONNECTED", 2: "CONFIRM_SAFETY", 3: "BOOTING",
               4: "POWER_OFF", 5: "POWER_ON", 6: "IDLE", 7: "BACKDRIVE", 8: "RUNNING"}
SAFETY_MODES = {1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
                6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", help="UR3e controller IP address")
    ap.add_argument("--config", default=None)
    ap.add_argument("--save", action="store_true", help="write configs/hardware.json")
    a = ap.parse_args()
    cfg = HardwareConfig.load(a.config, ip=a.ip)
    results = []

    def record(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"{OK if ok is True else (WARN if ok is None else BAD)}  {name:38s} {detail}")

    print(f"\nUR3e bring-up check — {cfg.ip}\n" + "=" * 72)

    # 1. network -------------------------------------------------------------
    ping = subprocess.run(["ping", "-c", "2", "-W", "2", cfg.ip], capture_output=True)
    record("network: ping", ping.returncode == 0,
           "" if ping.returncode == 0 else "no reply - check the cable, subnet and the robot's IP")
    for port, what in ((30004, "RTDE"), (29999, "dashboard")):
        try:
            with socket.create_connection((cfg.ip, port), timeout=2):
                record(f"port {port} ({what})", True)
        except Exception as e:
            record(f"port {port} ({what})", False, str(e))

    # 2. RTDE ----------------------------------------------------------------
    arm = None
    try:
        from hardware.devices import URArm
        arm = URArm(cfg).connect()
        record("RTDE connect", True, f"{cfg.rtde_frequency:.0f} Hz")
    except Exception as e:
        record("RTDE connect", False, f"{type(e).__name__}: {e}")
        print("\nCannot continue without RTDE.\n")
        return 1

    rm_, sm_ = arm.robot_mode(), arm.safety_mode()
    record("robot mode", rm_ == 8, f"{rm_} = {ROBOT_MODES.get(rm_, '?')}"
           + ("" if rm_ == 8 else "  -> power on and release the brakes on the pendant"))
    record("safety mode", sm_ == 1, f"{sm_} = {SAFETY_MODES.get(sm_, '?')}")

    # 3. state vs model ------------------------------------------------------
    q = arm.q()
    record("read joint angles", np.all(np.isfinite(q)), np.array2string(np.round(q, 4)))
    try:
        from core.kinematics import RobotModel
        rmod = RobotModel()
        p_model, _ = rmod.fk(q)
        p_robot = arm.tcp_pose()[:3]
        d = float(np.linalg.norm(p_model - p_robot))
        record("FK model vs robot TCP", d < 0.25,
               f"model {np.round(p_model,3)}  robot {np.round(p_robot,3)}  diff {1e3*d:.0f} mm"
               + ("  (expected: the model's TCP is the gripper tip, the robot's is the flange "
                  "unless a TCP offset is set on the pendant)" if d >= 0.02 else ""))
    except Exception as e:
        record("FK model vs robot TCP", False, f"{type(e).__name__}: {e}")

    # 4. gripper -------------------------------------------------------------
    # Probed, not assumed. Every probe is read-only: nothing here activates or actuates a gripper.
    from hardware.devices import detect_gripper
    found, probes = detect_gripper(cfg, arm)
    for pr in probes:
        record(f"probe: {pr['kind']}", True if pr["present"] else None, pr["detail"])

    if found == cfg.gripper:
        record(f"gripper ({cfg.gripper})", True, "configured gripper confirmed on the robot")
    elif found != "none":
        record("gripper", None,
               f"detected {found!r} but configs/hardware.json says {cfg.gripper!r}"
               + ("  -> --save will set it to the detected one" if a.save else
                  "  -> re-run with --save to adopt the detected one"))
        cfg.gripper = found
    elif cfg.gripper == "none":
        record("gripper", None,
               "none detected and none configured. If a gripper is fitted but not a Robotiq URCap, it is "
               "probably on a digital output: set gripper='digital_out' and gripper_do to its index.")
    else:
        record(f"gripper ({cfg.gripper})", False,
               f"configured as {cfg.gripper!r} but not detected - "
               + probes[0]["detail"])

    # Only now open the configured gripper, and only if one was actually found.
    if cfg.gripper != "none":
        try:
            from hardware.devices import make_gripper
            g = make_gripper(cfg, arm).connect(); g.close()
            record(f"gripper connect ({cfg.gripper})", True,
                   "connected" + (" (activation stroke done)" if cfg.gripper == "robotiq" else ""))
        except Exception as e:
            record(f"gripper connect ({cfg.gripper})", False, f"{type(e).__name__}: {e}")

    # 5. camera --------------------------------------------------------------
    if cfg.camera == "none":
        record("camera", None, "not configured")
    else:
        try:
            from hardware.devices import make_camera
            cam = make_camera(cfg).connect()
            t0 = time.time(); n = 0
            while time.time() - t0 < 1.0:
                rgb, depth = cam.read(); n += 1
            valid = float(np.mean(depth > 0))
            record(f"camera ({cfg.camera})", True,
                   f"{rgb.shape[1]}x{rgb.shape[0]}  {n} fps  depth valid {100*valid:.0f} %  "
                   f"median {np.median(depth[depth>0]):.3f} m")
            if cam.K_device is not None:
                print(f"         device intrinsics fx={cam.K_device[0,0]:.1f} fy={cam.K_device[1,1]:.1f} "
                      f"cx={cam.K_device[0,2]:.1f} cy={cam.K_device[1,2]:.1f}")
            record("depth coverage", valid > 0.5, f"{100*valid:.0f} % of pixels valid")
            cam.close()
        except Exception as e:
            record(f"camera ({cfg.camera})", False, f"{type(e).__name__}: {e}")

    # 6. calibration files ---------------------------------------------------
    from core.sensing import CFG_DIR
    for f, why in (("calibration.json", "run scripts/hw_calibrate.py"),
                   ("noise_model.json", "run scripts/hw_characterise_noise.py")):
        pth = os.path.join(CFG_DIR, f)
        exists = os.path.exists(pth)
        sim = exists and '"source": "simulation"' in open(pth).read()
        record(f"configs/{f}", exists and not sim,
               ("present" if exists else f"missing - {why}")
               + ("  (this is the SIMULATION file; it is not valid for the real camera - " + why + ")"
                  if sim else ""))

    arm.close()
    if a.save:
        print(f"\nwrote {cfg.save()}")
    nfail = sum(1 for _, ok, _ in results if ok is False)
    print("=" * 72)
    print(f"{len(results) - nfail} checks passed, {nfail} failed.  "
          + ("Ready for scripts/hw_identify_servo.py." if nfail == 0 else "Fix the failures before moving the robot."))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
