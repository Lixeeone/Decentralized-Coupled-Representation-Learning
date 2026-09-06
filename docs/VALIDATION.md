# Release validation record

Validation date: **2026-09-06**. These are measured release diagnostics, not transcribed manuscript results.

## Executed checks

| Check | Configuration | Observed result |
| --- | --- | --- |
| Scientific/reproduction tests | `python -m unittest discover -s tests -v` | **35 tests passed** |
| Editable installation | Local `pyproject.toml`, installed without dependency downloads | Passed |
| Wheel packaging | Offline wheel build, no dependency resolution | Passed |
| Release files | Local documentation links, original-source checksums, author-identifier scan | Passed |
| Synthetic suite | 9 configurations × seeds 0, 1, 2 | **27 completed**, 0 diverged |
| Mechanism ablations | 7 arms × seeds 0, 1, 2 | **13 completed**, **8 diverged**; all 21 runs recorded |
| Timescale sweep | 4 ratios × seeds 0, 1, 2 | **9 completed**, **3 diverged**; all 12 runs recorded |
| Circle graph generator | 4 sample sizes × 3 seeds × 2 schedules | 24 measured rows; complete grid coverage in all rows |
| Sphere graph generator | Same sample-size/seed/schedule design | 24 measured rows; 12 complete grids and 12 explicitly missing full-grid errors |
| Averaged flow | Known diagonal covariance, RK45 | Lyapunov decrease and principal-subspace convergence measured |
| Fixed-stream references | 5 methods × 3 seeds | 15 completed runs |
| Original supplementary runner | Supplied Swiss roll, HD-GMM and fixed-spatial gossip settings | Both expected diagnostic CSV files produced |
| Runtime primitive smoke check | N=256, n=64, m=4, 3 repetitions | All four timing primitives executed |
| Optional feature exporters | Source compilation and `--help` | Passed; neural model inference not executed |
| Missing feature archive | Configured ResNet input without data | Explicit file-missing error; no synthetic substitution |

The 35 tests cover covariance/rank-one update equivalence, spatial/plasticity order, no hidden QR, operator ablations, tangent/retraction constraints, the Lyapunov derivative and second-order Euler remainder, eigensolver agreement, Procrustes rotation invariance, nontrivial off-diagonal mass, rank-loss behavior, active-edge locality, gossip conservation, graph constants/Fourier modes/potential drift, feature checksums, unknown config rejection, overwrite protection, final-target correctness, and exact checkpoint continuation for both solvers.

## Measured quick-start endpoint

The recorded quick-start seed-0 run uses 256 HD-GMM observations in dimension 24, rank 2, 5,000 steps, `eta_x=0.2`, `eta_w=0.001`, `diffusion=0.0001`, and initial weight scale 0.8. There is no periodic or final QR in this coupled run. The endpoint target is recomputed from the final configuration.

| Quantity | Measured value |
| --- | ---: |
| `E_orth` | 1.2910062928e-7 |
| `E_sub` | 5.2838050170e-3 |
| `E_off_can` | 2.9434240469e-6 |
| Effective rank | 1.8607187836 |

The three-seed quick-start mean `E_sub` is approximately **0.009471**. Other synthetic configurations can retain substantial finite-horizon subspace error, particularly with weak/near-repeated spectral directions or asynchronous local updates. Successful program completion is not reported as universal convergence or reproduction of the paper's near-zero table entries.

The no-lateral-inhibition arm diverged in all three seeds. Scalar inhibition completed but its rank-collapse-aware `E_sub` was 1 in all three runs. Equal timescale steps diverged in all three seeds. Other step-size ratios showed non-monotone endpoint accuracy, so this validation does not assert that a smaller ratio always improves finite-horizon performance.

## Artifacts and commands

Measured records are under `examples/validation/`:

- `quickstart/`: full seed-0 state, diagnostics and figures.
- `synthetic/`, `ablations/`, `timescales/`: multi-seed endpoint/aggregate CSVs and each run's compact provenance/summary/history.
- `graph-circle/`, `graph-sphere/`: measured generator CSVs, environment and figures.
- `lyapunov/`: averaged-flow and Euler-remainder CSVs and figures.
- `baselines/`: fixed-stream results and provenance.
- `legacy/`: original reference diagnostics.
- `benchmark/`: compact timing measurements and units.
- `tests.txt`: executed test report.

The suite commands were:

```bash
python -m dcrl suite --config configs/synthetic_suite.yaml --output results/release-synthetic
python -m dcrl suite --config configs/ablations.yaml --output results/release-ablations
python -m dcrl suite --config configs/timescales.yaml --output results/release-timescales
python -m dcrl graph --config configs/graph_circle.yaml --output results/release-graph-circle
python -m dcrl graph --config configs/graph_sphere.yaml --output results/release-graph-sphere
python -m dcrl lyapunov --output results/release-lyapunov
python -m dcrl baselines --config configs/baselines.yaml --output results/release-baselines
python legacy/reference_diagnostics.py --output_dir results/legacy
python -m dcrl benchmark --samples 256 --dim 64 --rank 4 --repeats 3 --output results/release-benchmark
```

## Not executed

The 60,000/127,600-row pretrained-feature experiments, original 15,000-step paper-scale suites, GPU extraction, and original-hardware timing reproduction were not executed. The original feature/checkpoint archive was not supplied. The optional neural dependencies were not installed for the core validation. The GitHub workflow is provided but has not been run on a remote repository. The local README banner and measured plot images were visually inspected; no live GitHub page render is claimed.

All raw measurements are retained as observed. Divergence and uncovered graph queries remain visible. No downloaded pretrained feature is fabricated, no acceptance badge is asserted, and no manuscript number is inserted as a freshly computed output.
