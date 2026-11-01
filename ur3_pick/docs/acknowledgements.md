# Acknowledgements and third-party material

| Source | Use in this project | Licence |
| --- | --- | --- |
| [`SouthColumn76/universal_robots_ur3e`](https://github.com/SouthColumn76/universal_robots_ur3e) | UR3e MJCF description and meshes (`third_party_ur3e/`), copied unmodified to `ur3_pick/model/ur3e.xml` with two documented edits | BSD-3-Clause (`model/LICENSE_ur3e.txt`) |
| MuJoCo (DeepMind) | Physics engine, offscreen rendering, `mjd_transitionFD`, inverse dynamics | Apache-2.0 |
| OpenCV | Checkerboard detection, `calibrateCamera`, `calibrateHandEye`, connected components, `minAreaRect` | Apache-2.0 |
| SciPy / NumPy | `solve_discrete_are`, `expm`, rotations, NNLS variance fits | BSD |
| `python-control` | `ctrb` / `obsv` matrices for the rank analysis | BSD |
| 41014 weekly tutorials | Starting point for the EKF structure, the LQR formulation and the PBVS task description | Subject material |

**Edits to the third-party model** (both marked in a comment at the top of `model/ur3e.xml`):

1. the `keyframe` block was removed so the scene can add free-floating bodies;
2. the inner position servo's damping (`biasprm[2]`) was retuned from 200/100/40 to 42/24/9, i.e. to
   `2*sqrt(kp*M_ii)`, because the stock values saturate the joint torque limits above about 0.28 rad/s.
   Section "Failure analysis" of the project page explains how this was found.
