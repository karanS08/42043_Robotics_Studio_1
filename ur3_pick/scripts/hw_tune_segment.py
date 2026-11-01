"""Stage 3b: fit the cube's colour thresholds to the real camera and the real lighting.

    python scripts/hw_tune_segment.py --ip 192.168.1.102          # live frames from the wrist camera
    python scripts/hw_tune_segment.py --image frame.png           # an already-saved frame
    python scripts/hw_tune_segment.py --sim                       # a simulator frame (self-test)

The detector calls a pixel "cube" when it is bright red and much redder than it is green or blue. The
thresholds that do that in the simulator are wrong on real hardware: fluorescent tubes desaturate the
cube, a window behind it clips the red channel, and auto-exposure moves the whole histogram between
frames. Rather than guessing them at the bench, this finds them from the images themselves.

Method: locate the cube independently of the thresholds being fitted, using hue and saturation (a red
hue is a red hue whatever the exposure), take the largest such blob, then read the thresholds off that
blob's own statistics - a low percentile of its redness, minus a margin. Each candidate is then checked
the way the detector will use it: the mask must still select the cube, and must not leak into the
background. Writes configs/segment.json and saves previews so the mask can be eyeballed before a run.

NO MOTION. Sensors only; the control interface is never opened.
"""
import argparse, os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import cv2

from core.sensing import SegmentParams, segment_red, CFG_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "segment")

# Hue/saturation window used only to FIND the cube while the RGB thresholds are unknown.
HUE_TOL = 20          # degrees either side of pure red, in OpenCV's 0-179 hue scale this is +-10
SAT_MIN = 90
VAL_MIN = 60
MARGIN = 12           # how far below the blob's own 5th percentile to put each threshold


def find_cube_hsv(rgb: np.ndarray) -> np.ndarray:
    """Largest red blob, found by hue rather than by the thresholds being fitted."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    half = HUE_TOL // 2
    red = (((h <= half) | (h >= 180 - half)) & (s >= SAT_MIN) & (v >= VAL_MIN)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red)
    if n <= 1:
        return np.zeros(rgb.shape[:2], bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


def fit(frames: list, min_pixels: int = 25) -> tuple:
    """Thresholds from the cube's own pixel statistics, pooled over every frame."""
    reds, dg, db, found = [], [], [], 0
    for rgb in frames:
        m = find_cube_hsv(rgb)
        if m.sum() < min_pixels:
            continue
        found += 1
        px = rgb[m].astype(np.int16)
        reds.append(px[:, 0])
        dg.append(px[:, 0] - px[:, 1])
        db.append(px[:, 0] - px[:, 2])
    if not found:
        return None, 0
    r = np.concatenate(reds); g = np.concatenate(dg); b = np.concatenate(db)
    # 5th percentile: keep the dimmer edge pixels of the top face, not only its bright centre
    seg = SegmentParams(
        r_min=int(max(30, np.percentile(r, 5) - MARGIN)),
        r_minus_g=int(max(15, np.percentile(g, 5) - MARGIN)),
        r_minus_b=int(max(15, np.percentile(b, 5) - MARGIN)),
        open_ksize=3,
    )
    return seg, found


def score(frames: list, seg: SegmentParams) -> dict:
    """Check the fitted thresholds the way the detector will use them."""
    hits, iou, leak = 0, [], []
    for rgb in frames:
        ref = find_cube_hsv(rgb)
        got = segment_red(rgb, seg)
        if got.sum() >= 25:
            hits += 1
        if ref.sum() >= 25:
            inter = (ref & got).sum(); union = (ref | got).sum()
            iou.append(inter / max(1, union))
            leak.append(int((got & ~ref).sum()))
    return dict(detected=hits, n=len(frames),
                iou=float(np.mean(iou)) if iou else 0.0,
                leak_px=float(np.mean(leak)) if leak else 0.0)


def save_previews(frames: list, seg: SegmentParams):
    os.makedirs(OUT, exist_ok=True)
    for k, rgb in enumerate(frames[:4]):
        m = segment_red(rgb, seg)
        vis = rgb.copy()
        vis[m] = (0, 255, 0)                      # what the detector will call "cube"
        cv2.imwrite(os.path.join(OUT, f"frame{k}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, f"frame{k}_mask.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return OUT


def ascii_mask(rgb: np.ndarray, seg: SegmentParams, w: int = 56) -> str:
    """A look at the mask without needing a display."""
    m = segment_red(rgb, seg)
    h = max(1, int(w * m.shape[0] / m.shape[1] / 2))
    small = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
    return "\n".join("".join("#" if p else "." for p in row) for row in small)


# ---------------------------------------------------------------- frame sources
def frames_from_camera(cfg, n: int, warmup: int) -> list:
    from hardware.devices import make_camera
    cam = make_camera(cfg).connect()
    try:
        for _ in range(warmup):                   # let auto-exposure settle
            cam.read()
        out = []
        for _ in range(n):
            rgb, _ = cam.read()
            out.append(np.ascontiguousarray(rgb))
        return out
    finally:
        cam.close()


def frames_from_sim(n: int) -> list:
    from sim.env import Plant, PlantConfig
    from core.kinematics import RobotModel
    p = Plant(PlantConfig(seed=0))
    obs = p.reset()
    out, rm = [], RobotModel()
    for _ in range(n * 10):
        obs = p.step(p.truth.q, 0.0)
        if obs.rgb is not None and len(out) < n:
            out.append(np.ascontiguousarray(obs.rgb))
    p.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip")
    ap.add_argument("--image", help="a saved frame, or a glob of frames, instead of the camera")
    ap.add_argument("--sim", action="store_true", help="simulator frames (self-test of this script)")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--write", action="store_true", help="write configs/segment.json")
    a = ap.parse_args()

    if a.sim:
        frames = frames_from_sim(a.frames)
        src = "simulation"
    elif a.image:
        paths = sorted(glob.glob(a.image)) or [a.image]
        frames = []
        for p in paths:
            bgr = cv2.imread(p)
            if bgr is None:
                sys.exit(f"cannot read {p}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        src = f"images:{len(paths)}"
    else:
        from hardware.config import HardwareConfig
        cfg = HardwareConfig.load(ip=a.ip)
        print(f"camera={cfg.camera} {cfg.width}x{cfg.height} @{cfg.fps}  (no motion; sensors only)")
        frames = frames_from_camera(cfg, a.frames, a.warmup)
        src = "hardware"

    print(f"{len(frames)} frames from {src}\n")
    old = SegmentParams.load()
    print(f"current : {old}   ->  {score(frames, old)}")

    seg, found = fit(frames)
    if seg is None:
        print("\nNo red blob found in any frame by hue either. Nothing to fit from.\n"
              "  - is the cube in view? check results/segment/frame0_rgb.png\n"
              "  - is the lighting so warm that the cube's hue is no longer red?\n"
              "  - is the camera returning BGR instead of RGB?")
        save_previews(frames, old)
        sys.exit(1)

    sc = score(frames, seg)
    print(f"fitted  : {seg}   ->  {sc}")
    print(f"\ncube located by hue in {found}/{len(frames)} frames")
    print(f"mask agrees with the hue blob: IoU {sc['iou']:.2f}, background leak {sc['leak_px']:.0f} px/frame")
    print("\nmask the detector would use (fitted thresholds):\n")
    print(ascii_mask(frames[0], seg))

    d = save_previews(frames, seg)
    print(f"\npreviews: {d}/frame*_mask.png  (green = called cube)")

    if sc["detected"] < len(frames):
        print(f"\nWARNING: fitted thresholds still miss the cube in "
              f"{len(frames) - sc['detected']}/{len(frames)} frames.")
    if sc["iou"] < 0.5:
        print("\nWARNING: IoU below 0.5 - the mask and the hue blob disagree. Look at the previews "
              "before trusting this; the largest red thing in view may not be the cube.")

    if a.write:
        p = seg.save(source=src, n_frames=len(frames), iou=round(sc["iou"], 3),
                     leak_px=round(sc["leak_px"], 1))
        print(f"\nwrote {p}")
    else:
        print("\n(nothing written - re-run with --write to save these thresholds)")


if __name__ == "__main__":
    main()
