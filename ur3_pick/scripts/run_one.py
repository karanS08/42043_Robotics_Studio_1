"""Run a single pick trial and print the metrics.  Usage: python scripts/run_one.py [seed]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
from eval.runner import TrialConfig, run_trial

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    r = run_trial(TrialConfig(seed=seed))
    for k, v in r.items():
        print(f"{k:26s} {v}")
