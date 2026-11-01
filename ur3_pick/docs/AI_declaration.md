# Generative AI declaration

**Tool:** Claude (Anthropic), used through Claude Code in the terminal.

**Purpose and extent of use.** The assistant was used as a pair programmer and reviewer throughout the
project. Specifically it was used to:

* scaffold the repository layout and the MuJoCo scene that extends the third-party UR3e description
  (table, cube, wrist RGB-D camera, two-finger gripper);
* write first drafts of the Python modules (`sim/`, `core/`, `eval/`, `scripts/`, `tests/`);
* help diagnose four defects found during development, listed in the failure analysis of the project page
  (servo damping/LQR instability, detector locking onto the tool marker, biased cube-height estimator,
  over-confident object covariance);
* draft the text of this project page and the figures' captions.

**What was not delegated.** The choice of track and scope, the modelling assumptions, the decision of which
quantities to measure, the acceptance criteria for each part, and the interpretation of the results are the
author's own. Every number quoted on the project page is produced by the scripts in this repository and can be
regenerated with the commands in the README; no result was taken from the assistant's description of a result.

**Verification.** All generated code was executed and checked against ground truth from the simulator
(`eval/`), against analytic expectations (linear-model validation in `scripts/model_analysis.py`), and against
filter-consistency statistics (NEES/NIS). The ground-truth firewall is enforced mechanically by
`tests/test_firewall.py` rather than by inspection.
