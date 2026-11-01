"""Convert the result videos to GIFs so they play inline on the GitHub project page.

    .venv/bin/python scripts/make_gifs.py            # all of them
    .venv/bin/python scripts/make_gifs.py nominal moving

Why GIFs and not the mp4s that are already in results/video/: GitHub does not render a video player for
an mp4 committed in a repository. Neither a <video> tag nor a raw link works - only files uploaded through
the web UI (which land under github.com/user-attachments/assets/...) get a player, and those cannot be
produced from a script. An animated GIF referenced with ordinary image syntax plays inline everywhere,
including in a local clone and in the offline Markdown, so that is what the README uses.

Uses the ffmpeg binary that ships inside imageio-ffmpeg, so there is nothing extra to install. Two passes:
palettegen builds an optimal 128-colour palette for each clip, paletteuse applies it with ordered
dithering. A single global palette would band the gradients on the table badly.
"""
import os
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "video")
DST = os.path.join(ROOT, "results", "gif")

WIDTH = 640          # half the 1240 px render; the telemetry panel is still readable
FPS = 10             # from 25; the motion is slow enough that 10 reads the same
COLORS = 128

FILTER = (f"fps={FPS},scale={WIDTH}:-1:flags=lanczos,split[s0][s1];"
          f"[s0]palettegen=max_colors={COLORS}[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3")


def convert(exe: str, name: str) -> tuple:
    src = os.path.join(SRC, name + ".mp4")
    dst = os.path.join(DST, name + ".gif")
    if not os.path.exists(src):
        return name, None, f"missing {src}"
    r = subprocess.run([exe, "-y", "-v", "error", "-i", src, "-vf", FILTER, "-loop", "0", dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return name, None, r.stderr.strip().splitlines()[-1] if r.stderr else "ffmpeg failed"
    return name, os.path.getsize(dst), None


def main():
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(DST, exist_ok=True)
    wanted = sys.argv[1:]
    names = wanted or sorted(f[:-4] for f in os.listdir(SRC) if f.endswith(".mp4"))
    if not names:
        sys.exit(f"no .mp4 found in {SRC} - run scripts/make_video.py first")

    print(f"ffmpeg: {exe}\n{WIDTH}px, {FPS} fps, {COLORS} colours -> {DST}\n")
    total, failed = 0, 0
    for n in names:
        name, size, err = convert(exe, n)
        if err:
            print(f"  FAIL  {name:24s} {err}")
            failed += 1
        else:
            total += size
            print(f"  ok    {name:24s} {size/1e6:.2f} MB")
    print(f"\n{len(names) - failed}/{len(names)} converted, {total/1e6:.1f} MB total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
