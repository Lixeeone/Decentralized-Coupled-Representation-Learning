# Reproduction protocol

## Environment

The core environment is pinned in `requirements.txt` and validated on Python 3.12. The broader package ranges in `pyproject.toml` express compatibility intent; only environments explicitly listed in `VALIDATION.md` were executed. Install the pinned requirements before installing the local package. Core experiments require NumPy, SciPy, Matplotlib, PyYAML and threadpoolctl. No neural framework is required.

The CLI limits numerical-library threads to one by default. To change this, put the global option before the subcommand:

```bash
python -m dcrl --threads 4 run --config configs/quickstart.yaml --output results/four-threads
```

Changing hardware, numerical libraries or thread counts can change floating-point trajectories. Equal seeds provide repeatability in the same environment; the project does not claim cross-platform bitwise equality. `metadata.json` records Python/package versions, BLAS details, source fingerprints, and input hashes without dumping user-specific environment variables.

## Run protocol

1. Work from the repository root. Relative feature paths are resolved from the working directory. Suite job configuration paths are resolved relative to the suite YAML file.
2. Inspect the selected YAML configuration. `protocol_origin` states whether settings were supplied or reconstructed.
3. Run the command into a new output directory. The runner rejects existing outputs unless explicit checkpoint resume is requested.
4. Check `summary.json`: completion status, requested/completed steps, target protocol, eigengap, eigen residual, and raw endpoint metrics.
5. Use `history.csv` for trajectory plots and `aggregate.csv` for multi-seed statistics. Never copy printed manuscript values into result files.

The code uses local NumPy generators. Initialization has an independent deterministic seed offset; spatial noise and gossip activation share the run generator. Preprocessing is fit once on the selected initial feature matrix and saved in metadata. The graph evaluation grid is deterministic and fixed across sample sizes and seeds.

## Checkpoints

The checkpoint includes the initial configuration, current state, RNG state, step counter, diagnostic history and configuration/source fingerprints. A-DCRL additionally saves every local representation copy. Arrays and JSON metadata use `np.savez_compressed`; no Python pickle objects are loaded.

Resume requires the same data specification, dynamics, seed, target protocol and source implementation. The horizon can increase; it must not be smaller than the saved step. Output location, logging frequency and plot generation can change. Exact state continuation for synchronous and asynchronous runs is tested against uninterrupted execution. Changing logging frequency changes the sampled diagnostic history, not the training updates.

Checkpoint creation uses a temporary file and atomic replacement in the same directory. Checkpointing is synchronous, so it adds I/O cost to a run. A resumed run uses its checkpoint's stored data, not a newly loaded feature file; its original input hash is retained.

## Failure and missing-value handling

Nonfinite states or the configured weight-norm threshold terminate a run with `status: diverged`. Its last finite measurements are labeled `last_finite_before_divergence` and are not presented as successful endpoints. The standalone run command returns exit code 2. A suite continues other seeds/arms and records explicit divergence counts.

Suite mean/std columns include completed runs only, with the number contributing to each metric. Undefined metrics are excluded and counted. Standard deviations use `ddof=1`; a single run has no sample standard deviation. Undefined scalars use JSON `null` or empty CSV cells, never fabricated zeroes.

The orthogonality-violation diagnostic records all synchronous steps. It is a finite-step numerical measure, not a proof of monotonicity of a stochastic coupled trajectory. It is not applied as an averaged-flow theorem to the mean of local A-DCRL copies.

## Graph diagnostic

The graph kernel uses geodesic-radius neighborhoods on the unit circle or sphere. Angular/geodesic radii are converted to chord radii solely for KD-tree queries. Self transitions are excluded. Gibbs weights are normalized after a numerically stable logit shift. The jump rate is exactly $2D(d+2)/\epsilon_N^2$.

Four smooth functions are evaluated on the same finite grid for both empirical and analytic operators. On the circle these are sine/cosine modes 1 and 2 with potential `amplitude * cos(theta)`. On the sphere they are `x`, `z`, `x²`, `z²` with potential `amplitude * z`. `E_L_grid` is a finite-grid/test-family maximum, not a supremum over every point and every function.

If even one evaluation query has an empty neighborhood, the complete-grid error is undefined; coverage and degree statistics are still exported. A violated scaling regime can fail in this way. The code never drops uncovered points and reports a smaller error as though the full grid had been evaluated.

## Cost and timing

`dcrl benchmark` times explicitly labeled primitives with a warmup and repeated measurements. Dense PCA reports one covariance construction/eigensolve; streaming Oja reports one full sample pass; DCRL reports one full synchronous spatial/plasticity step; A-DCRL reports one local active-edge primitive. These units cannot be divided to obtain an end-to-end algorithm speedup. The timed A-DCRL primitive excludes allocating the full simulator state.

The synchronous solver stores `X`, `W` and temporary observation blocks. A-DCRL stores all local copies in its single-process simulation, which can be expensive for large `N`, `m`, and `n`. Exact PCA deliberately allocates a dense covariance. The default evaluation solver uses a matrix-free `LinearOperator` for larger ambient dimensions and reports its eigen residual and cutoff eigengap.

## Release examples

`examples/validation/` contains measured validation artifacts, with the command record in `docs/VALIDATION.md`. Only the quick-start example includes its full checkpoint/final arrays; other examples retain compact logs, summaries and selected figures. Fresh runs write to the ignored `results/` directory and do not overwrite the release examples.

## Publication preparation

The archive is ready to initialize as a GitHub repository and includes a CI workflow, issue template, citation placeholder, and MIT code license. It contains no `.git` history and makes no claim that a remote repository or GitHub Actions run already exists. Git commit identity and the remote account will be determined by the account used to publish it. Complete `CITATION.bib` only when publication metadata should become public.
