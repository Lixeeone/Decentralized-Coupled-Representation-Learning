# Mathematical conventions and implementation

## Two update phases

Rows of `X` are observations, so `X.shape == (N, n)` and `W.shape == (m, n)`. Algorithm 1 first freezes `W` and uses

$$g_i=\Pi_{x_i}W^\top Wx_i,\quad \xi_i=\Pi_{x_i}\widetilde\xi_i,\quad \widetilde\xi_i\sim\mathcal N(0,I).$$

The spatial update is

$$x_i^+=\Pi_{\mathcal X}\!\left(x_i+\eta_xD\beta g_i+\sqrt{2D\eta_x}\xi_i\right).$$

The drift sign follows the negative gradient of $U_W(x)=-\frac12\|Wx\|^2$. Plasticity then uses `X_new`, not the old configuration:

$$Y=X_{\mathrm{new}}W^\top,\quad F(W)=N^{-1}Y^\top(X_{\mathrm{new}}-YW).$$

This is algebraically equivalent to $W\widehat\Sigma-W\widehat\Sigma W^\top W$. A block implementation bounds observation workspace and avoids dense training covariance construction. Storage of the configuration itself still costs $O(Nn)$; $O(mn)$ describes one representation copy, not total simulator memory.

The core uses `float64`. `eta_w` already contains the discretized product $\epsilon\gamma\Delta t$; no extra hidden plasticity multiplier is applied. Both drift and noise are zero when `diffusion=0`. To remove noise while retaining drift, an explicitly different deterministic solver would be needed; the averaged-flow diagnostic separately integrates the fixed-covariance ODE.

## Geometric scope

| Domain | Tangent operation | Retraction | Interpretation |
| --- | --- | --- | --- |
| Sphere | Remove radial component | Normalize to radius | Compact boundaryless manifold |
| Torus | Remove analytic surface-normal component | Project to torus with major radius 2, minor radius 0.7 | Compact boundaryless manifold |
| Swiss roll / S-curve | Project to curve tangent plus height direction | Coarse parameter search with safeguarded Newton refinement; clamp height | Parametric strip with boundary; numerical stress protocol |
| Cylinder / hyperboloid | Remove analytic normal component | Radial/axial projection, clipped height | Bounded manifold patch with boundary |
| Empirical ball | Ambient identity tangent | Clip around the initial empirical mean, using its maximum sample radius | Controlled Euclidean diagnostic domain |
| Identity | Identity | Identity | Fixed-stream experiments only |

The torus sampler is area-uniform by rejection sampling; sampling its two angles uniformly would not be area-uniform. The circle and sphere graph diagnostics use uniform geometric sampling. Gaussian mixtures, sparse clouds, and intersecting planes are controlled distributions and do not satisfy the compact smooth boundaryless theorem assumptions merely because their simulations are bounded.

## Diagnostic definitions

Let $\widehat\Sigma=X^\top X/N$, $\Sigma_Y=W\widehat\Sigma W^\top$, and let $Q_m$ contain the selected target's top eigenvectors. The matrix called covariance in the manuscript is a second moment. Initial centering/scaling is explicit in configuration; subsequent metric calls do not recenter `X`.

| Metric | Definition and convention |
| --- | --- |
| `E_orth` | $\|WW^\top-I_m\|_F$, measured on raw weights |
| `V_orth` | $\frac14 E_{\mathrm{orth}}^2$ |
| `E_sub` | Largest principal-angle sine; the spectral norm of $(I-Q_mQ_m^\top)Q_W$ |
| `E_off_can` | $\|\operatorname{offdiag}(R^\top\Sigma_YR)\|_F/\|\Sigma_Y\|_F$ |
| `effective_rank` | $\exp(-\sum_jp_j\log p_j)$, normalized nonnegative latent eigenvalues $p_j$ |
| `variance_explained` | Fraction of input second-moment energy captured by an orthonormal basis of `Row(W)` |
| `consensus_error` | Square root of mean squared Frobenius disagreement of local copies from their population mean |

For the canonical gauge, compute $WQ_m=ASB^\top$ and set $R=AB^\top$. This alignment is determined by the target eigenbasis. It does not diagonalize $\Sigma_Y$ as a separate operation. If `W @ Q_m` is numerically rank-deficient, the gauge is not uniquely determined and `E_off_can` is undefined. Zero latent energy is also undefined, not perfect decorrelation.

Rank-deficient `W` receives `E_sub=1`. QR stabilization refuses rank collapse instead of filling in unobserved directions. The residual-based angle computation avoids cancellation in `sqrt(1 - smallest_cosine**2)` near convergence. Repeated target eigenvalues can make the cutoff subspace non-unique; each endpoint records its cutoff eigengap and eigen residual. No result is forcibly set to zero to match printed table precision.

## Target protocols

`target: final` is the default: logged trajectories use the initial eigenspace, while endpoint diagnostics recompute a reference from the final configuration. This makes trajectory evaluation possible online and matches the supplied reference's final-configuration endpoint rule. The two targets are labeled separately in artifacts and plot captions. They need not give the same last plotted and endpoint values.

`target: initial` uses the initial eigenspace throughout. `target: current` recomputes the eigenspace at each logging step and at the endpoint; it can be expensive on high-dimensional data. Target eigenvectors are used only for measurement and are never passed into a training update.

## Stability and claims

For the averaged ODE $\dot W=W\Sigma-W\Sigma W^\top W$, the code tests

$$\frac{d}{d\tau}V_{\mathrm{orth}}=-\operatorname{Tr}\left[(WW^\top-I)^2W\Sigma W^\top\right]\le0.$$

The independent Euler test subtracts this first-order derivative and reports the measured remainder divided by $\eta_w^2$. Finite-step coupled runs can show positive increments. The runner records every synchronous step's increment, not just logged points, and uses `delta_V > 1e-12 * (1 + abs(V_before))` for its disclosed violation count. These statistics are release measurements; their threshold was not specified by the original archive.

The graph limit assumes a fixed smooth potential and geometric scaling. The averaging/alignment results concern the fixed or frozen invariant covariance under the stated hypotheses. The moving-configuration implementation also realizes the quasi-static coupled comparison described in the paper. Arbitrary finite-step hyperparameters and stochastic trajectories are not guaranteed to inherit deterministic pointwise energy descent.

## A-DCRL

One oriented edge `(i,j)` is sampled uniformly from ring orientations, or uniformly from the complete graph's ordered distinct pairs. Endpoint `i` first updates its spatial state under `W_i`, then performs a rank-one Oja update. Pairwise mixing applies to the updated `W_i` and the old `W_j`. Only these two matrices and spatial state `x_i` change. The mixing component preserves the input pair's average; the preceding local Oja increment can change the population average.

Global averaging and disagreement calculations occur only when diagnostics/checkpoints are requested. The loop's local update uses neither a global covariance nor inactive agents. Total single-process local-copy storage is $O(Nmn)$; per-agent storage is $O(mn+n)$. Two directions of a matrix exchange carry two $mn$-element payloads, if both directions are counted.

## Fixed-stream references

The references are explicit implementations, not imported reproductions of unspecified manuscript baseline code. `streaming_oja` uses full matrix subspace inhibition; `sanger` uses lower-triangular GHA inhibition. Both use a shared seeded minibatch order and configurable periodic QR. `eigengame` uses an alpha-EigenGame-type parent-penalty gradient with unit-row normalization. `distributed_orthogonal_iteration` sums local covariance actions across shards and performs synchronous QR; it is not an implementation of DeEPCA or a pairwise-gossip eigensolver. Final QR is disclosed for all iterative fixed-stream references.

Primary technical references: [EigenGame paper](https://arxiv.org/abs/2010.00554), [SciPy symmetric iterative eigensolver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html), and [TorchVision model/weight documentation](https://docs.pytorch.org/vision/stable/models.html).
