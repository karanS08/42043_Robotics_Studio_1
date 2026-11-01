# Reproduce the whole project. Every target runs inside a memory-capped scope (scripts/capped.sh).
PY      ?= ../.venv/bin/python
CAP     ?= scripts/capped.sh --mem 5G
WORKERS ?= 8
MC      ?= 50
TUNE    ?= 15
export MUJOCO_GL ?= egl

.PHONY: all calib noise model tune mc plots report video gifs test clean \
        hw-install hw-check hw-identify hw-calib hw-segment hw-noise hw-dry hw-run hw-compare
all: calib noise model tune mc plots report

calib:  ; $(CAP) $(PY) scripts/calibrate.py
noise:  ; $(CAP) $(PY) scripts/characterise_noise.py
model:  ; $(CAP) $(PY) scripts/model_analysis.py
tune:   ; UR3_WORKERS=$(WORKERS) $(CAP) $(PY) scripts/tuning.py $(TUNE)
mc:     ; UR3_WORKERS=$(WORKERS) $(CAP) $(PY) scripts/monte_carlo.py $(MC)
plots:  ; $(CAP) $(PY) scripts/make_plots.py && $(CAP) $(PY) scripts/make_frame_diagram.py
gifs:   ; $(PY) scripts/make_gifs.py
report: ; $(PY) scripts/make_report.py
video:  ; $(CAP) $(PY) scripts/make_video.py
test:   ; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= $(PY) -m pytest

clean:  ; rm -f results/fig_*.png results/mc_*.json results/tuning_*.json docs/RESULTS.md

# ---------------------------------------------------------------- real hardware
# Pass the robot address once:  make hw-check IP=192.168.1.102
IP ?= 192.168.1.102
hw-install: ; $(PY) -m pip install -r requirements-hw.txt
hw-check:   ; $(PY) scripts/hw_check.py --ip $(IP) --save
hw-identify:; $(PY) scripts/hw_identify_servo.py --ip $(IP)
hw-calib:   ; $(PY) scripts/hw_calibrate.py --ip $(IP)
hw-segment: ; $(PY) scripts/hw_tune_segment.py --ip $(IP) --write
hw-noise:   ; $(PY) scripts/hw_characterise_noise.py --ip $(IP)
hw-dry:     ; $(PY) scripts/hw_run.py --ip $(IP)
hw-run:     ; $(PY) scripts/hw_run.py --ip $(IP) --move
hw-compare: ; $(PY) scripts/hw_compare.py $(LOG)
