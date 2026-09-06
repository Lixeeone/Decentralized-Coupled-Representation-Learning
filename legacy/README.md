# Anonymous Reference Implementation for DCRL

This package provides a compact anonymous reference implementation for the update rules and diagnostic metrics used in the paper.

It includes:

- synchronous DCRL;
- asynchronous pairwise-gossip A-DCRL;
- self-contained reference substrates;
- invariant diagnostics for row-orthogonality, subspace alignment, and covariance decorrelation.

The purpose of this package is to make the algorithmic updates and diagnostic computations transparent. It is not intended to reproduce every large-scale table or pretrained-feature experiment in the paper.

## Structure

```text
dcrl_reference_implementation_anonymous/
  README.md
  requirements.txt
  run_reference_diagnostics.sh
  reference_diagnostics.py

  src/
    data.py
    dcrl_sync.py
    adcrl_gossip.py
    metrics.py
    utils.py
```

## Setup

```bash
pip install -r requirements.txt
```

The implementation uses lightweight numerical dependencies only. No external datasets or pretrained model weights are required for the reference diagnostics.

## Running the reference diagnostics

```bash
bash run_reference_diagnostics.sh
```

This runs compact diagnostics for synchronous DCRL and asynchronous A-DCRL on self-contained reference substrates. Outputs are written to:

```text
results/
```

Typical output files include:

```text
results/sync_dcrl_diagnostics.csv
results/adcrl_gossip_diagnostics.csv
```

## Implemented components

### Synchronous DCRL

The synchronous implementation follows the two-phase update in the paper:

1. a projected spatial step under the frozen representation-induced potential;
2. a finite-sample subspace Oja update with lateral inhibition.

In particular, the spatial drift uses the projected direction

```text
Pi_x(W.T @ W @ x)
```

corresponding to the negative gradient direction of the potential
`U_W(x) = -0.5 * ||W x||^2`.

### Asynchronous A-DCRL

The asynchronous implementation uses pairwise active-edge gossip. Each agent maintains a local representation matrix. At each communication round, one active edge is selected, a local Oja update is applied at the active endpoint, and the two endpoint matrices are mixed by pairwise averaging. Inactive agents are unchanged.

This implementation is intended to reflect the local communication footprint described in the paper.

### Diagnostics

The package implements the following diagnostic quantities:

- row-orthogonality error `E_orth`;
- principal-subspace alignment error `E_sub`;
- canonical-gauge covariance off-diagonal mass `E_off_can`.

These diagnostics correspond to the invariant-level quantities used in the numerical verification section of the paper.

## Scope

This package is a reference implementation, not a full experiment archive. In particular, it does not include:

- pretrained feature matrices;
- large-scale cached outputs;
- full sweep logs;
- hardware-specific timing runs;
- non-anonymous project files.

The same routines can be applied to large-scale feature experiments by replacing the generated reference inputs with an external feature matrix of shape `(N, n)`.

## Anonymity

The package is prepared for anonymous review. It contains no author names, institutional identifiers, usernames, machine-specific absolute paths, API keys, checkpoints, or external credentials.
