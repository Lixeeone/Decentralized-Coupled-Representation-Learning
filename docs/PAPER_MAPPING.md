# Paper-to-code mapping and provenance

This release was constructed from the supplied manuscript and its compact supplementary source. It does not contain author-identifying originals. `legacy/SOURCE_MANIFEST.json` records the extracted source files after line-ending normalization; their mathematical code is unchanged.

## Coverage

| Manuscript item | Runnable implementation | Provenance and limit |
| --- | --- | --- |
| Algorithm 1; Eqs. D.1–D.4 | `dcrl.dynamics`, `dcrl.geometry` | Equations implemented, with explicit numerical retractions |
| Algorithm 2; Appendix D.2 | `dcrl.gossip` | Full local spatial/Oja/mixing sequence; single-process simulation |
| Eqs. 2–5; Theorem 1 | `dcrl graph` | Circle/sphere Gibbs graph tests with analytic reference operators; reconstructed test grids and potential |
| Eq. 11; Theorem 2 | `dcrl lyapunov` | Fixed-covariance ODE plus independent derivative/finite-step tests |
| Theorems 3 and 5; Section 5 | `dcrl.metrics`, synthetic run configurations | Principal-angle and Procrustes-gauge diagnostics |
| Figure 2's 15,000-step spectral horizon | `configs/paper_scale/`, `spectrum.svg` | Horizon retained; original trajectories unavailable |
| Table 1 / D.5 geometry and dimensions | `configs/paper_scale/`, `configs/features/` | Stated sizes/ranks retained; not exact numeric table regeneration |
| Table 2 / D.6 operator ablations | `configs/ablations.yaml` | Full, static-X, frozen-W, no/scalar inhibition, and large-step arms |
| Table D.7 timescale sweep | `configs/timescales.yaml` | Ratios 0.005, 0.05, 0.2, plus equal-timescale reference; reconstructed absolute steps |
| Table D.4 fixed-stream calibration | `dcrl baselines`, feature baseline configs | Rank 8, up to 20,000 samples, three seeds; explicitly specified replacement baseline schedules |
| Table 3 / D.2 resource primitives | `dcrl benchmark` | Re-measured primitive timings with units and environment; no copied timing values |
| Supplied reference outputs | `legacy/reference_diagnostics.py` | Original Swiss-roll/HD-GMM ball-domain and fixed-spatial gossip reference routines |

## What was supplied

The supplementary source includes a small NumPy reference runner, two synthetic reference cases, a fixed-spatial pairwise-gossip routine, three invariant metrics, initialization, and bounded ball retraction. It explicitly describes itself as a reference implementation rather than a full experiment archive.

The manuscript additionally specifies tangent-projected spatial drift/noise and actual substrate retractions for embedded manifolds. Its Algorithm 2 includes a local spatial step absent from the compact gossip reference. Those mechanisms are implemented in the main package while the supplied version remains available under `legacy/`.

## Explicit reconstruction choices

The main package's default settings are not presented as recovered experimental metadata. New choices include random seeds `0,1,2`; finite-step schedules where absent; grid sizes and smooth graph test functions; precise synthetic sampling distributions; initialization scale; empirical-ball radius policy; baseline normalization schedules; eigensolver tolerance; and the finite-step violation threshold. Configurations contain a `protocol_origin` field where applicable.

The manuscript's Appendix D describes bounded empirical domains but does not provide the per-run radius archive. This release derives each ball's center and maximum sample radius from the prepared initial input. Manifold configurations preserve their native coordinates and reject centering/scaling that would invalidate analytic tangent formulas.

## Missing inputs for exact manuscript tables

Exact table-level reproduction still requires all of the following:

1. Original feature matrices, or exact encoder checkpoints/revisions, extraction layers, transforms, tokenization, pooling and row order.
2. Complete per-experiment step sizes, schedules, initialization, seeds and domain parameters.
3. Original graph diagnostic functions, neighborhood radii, evaluation grids and aggregation details.
4. The baseline implementations/schedules, complete raw outputs and original timing environment.

The supplied manuscript also restricts generator diagnostics to sampled manifold graph processes in its protocol text, while some synthetic-cloud table entries display generator values. This release computes `E_L_grid` only where it has an explicit graph and an analytic reference generator. It does not assign graph-generator errors to an arbitrary feature cloud.

No manuscript result is injected into a CSV or plotted as a fresh experiment. Archived release examples identify their settings and validation status. Once original experimental inputs become available, the existing config/data interfaces can be used to regenerate and compare results transparently.
