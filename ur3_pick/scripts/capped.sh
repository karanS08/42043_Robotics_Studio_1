#!/usr/bin/env bash
# Run any command inside a memory-capped systemd scope so a runaway job cannot take the laptop down.
# Usage: scripts/capped.sh [--mem 4G] <command...>      (default cap 4G, no swap; the kernel kills the job, not the desktop)
MEM=4G
if [ "$1" = "--mem" ]; then MEM="$2"; shift 2; fi
exec systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 \
  env MUJOCO_GL="${MUJOCO_GL:-egl}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"
