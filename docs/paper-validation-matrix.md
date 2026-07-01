# Paper Validation Matrix

| Area | Paper or Source Evidence | Local Level 2 Target | Current Status |
|---|---|---|---|
| Bibliography | Title, authors, arXiv `2203.11203`, DOI `10.1016/j.neunet.2022.10.022`, *Neural Networks* 157:288-304 | Cite stable metadata and avoid distributing the PDF | Metadata captured |
| Problem domain | Automatic 2D quadrilateral mesh generation for CAD/E-style simulation preprocessing | Support 2D polygon boundaries and quadrilateral outputs | Planned/implemented locally |
| Non-domain | Paper is not a 3D surface/volume meshing system | Keep 3D out of scope | Explicit non-goal |
| MDP framing | Boundary evolves as one quadrilateral is extracted per step | Provide environment with state, action, reward, done/truncated conditions | Level 2 scaffold |
| Reference vertex | Paper chooses the local reference vertex by minimum averaged surrounding boundary angle | Include local-boundary features and deterministic metadata where practical | Approximate local implementation |
| State | Paper uses partial observation around reference vertex plus progress ratio | Fixed-size numeric observation for SAC-compatible smoke training | Approximate local implementation |
| Action | Paper considers action types adding zero, one, or two vertices, with final article using `[type, V1]`; type 2 is rare | Continuous action space for smoke training, documented as proxy if not exact | Approximate local implementation |
| Reward | Paper uses invalid `-0.1`, terminal `10`, otherwise element quality + boundary quality + density | Proxy reward using completion, valid quads, element quality, and invalid penalties | Not claimed exact |
| SAC | Paper selects SAC; hidden layers `[128,128,128]`, batch `256`, gamma `0.99`, learning rate `3e-4`, total steps `1.2e6` | Local config mirrors recovered hyperparameters and supports CPU smoke/GPU run | Configured |
| Training domain | Paper selects `T1` after comparing `T1/T2/T3` | Generate diverse synthetic local cases; optionally inspect upstream domains later | Synthetic first |
| Evaluation domains | Paper tests scalability, generalizability, quality; domains include sharp angles, bottlenecks, uneven boundaries, holes | Cover simple convex, concave, narrow-channel, multiscale, noisy, near-degenerate cases | Synthetic first |
| Baselines | Paper compares Blossom-Quad and Pave | Use a local deterministic baseline for Level 2 only | Not paper-parity |
| Metrics | Paper uses singularity, element quality, angle deviations, scaled Jacobian, stretch, taper, triangle count | Implement completion, valid quad ratio, invalid count, Eq. 7 element quality, angle/aspect/runtime, and clearly named substitutes | Local metrics |
| Public code | `design-zeng/ReinforcementLearning4MeshGeneration` exists, MIT, no releases/tags/checkpoints observed | Treat as reference and optional future comparison source | Not vendored |
| Artifacts | No official checkpoint/release found in public code pass | Do not claim pretrained model availability | Documented |
| Current goal | Local Level 2 plus GPU-ready package | Do not claim Level 3 until GPU training/evaluation is run | Active guardrail |

## Recovered Paper Metric Targets

The paper reports Table 6 quantitative comparisons averaged over domains 7-9. These are **not local targets for the current goal**, but they define later Level 3 comparison context:

| Metric | Blossom-Quad | Pave | FreeMesh-RL |
|---|---:|---:|---:|
| Singularity, lower better | 388 +/- 209.50 | 146.70 +/- 51.50 | 132 +/- 50 |
| Element quality, higher better | 0.72 +/- 0.12 | 0.79 +/- 0.12 | 0.79 +/- 0.13 |
| `|MinAngle - 90|`, lower better | 6.55 +/- 6.91 | 3.69 +/- 4.60 | 4.02 +/- 5.10 |
| `|MaxAngle - 90|`, lower better | 22.16 +/- 11.14 | 15.69 +/- 14.71 | 15.73 +/- 12.48 |
| Scaled Jacobian, higher better | 0.91 +/- 0.08 | 0.94 +/- 0.13 | 0.94 +/- 0.10 |
| Stretch, higher better | 0.79 +/- 0.08 | 0.84 +/- 0.10 | 0.83 +/- 0.11 |
| Taper, lower better | 0.15 +/- 0.11 | 0.12 +/- 0.14 | 0.11 +/- 0.11 |
| Triangle count, lower better | 2.70 +/- 2.50 | 8 +/- 2.80 | 0 +/- 0 |

## Open Validation Items

- Visually inspect PDF tables before a Level 3 run to confirm all hyperparameters and domain values.
- Map paper domains `T1` and `D7-D9` to public repository `ui/domains/*.json` files, if possible.
- Decide whether to port selected upstream domain JSON files as small examples after confirming license and attribution.
- Validate scaled Jacobian, stretch, taper, and singularity definitions against Verdict/Gmsh or the upstream implementation before reporting paper-style values.
