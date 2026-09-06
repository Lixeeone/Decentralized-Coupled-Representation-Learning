# Data and pretrained feature inputs

## Core synthetic inputs

The numerical experiments generate their own inputs and need no downloads. Seeds govern both data and initialization. The reference Swiss roll/HD-GMM generators are preserved under `legacy/src/data.py`. The main package adds sphere, area-uniform torus, S-curve, cylinder, truncated hyperboloid, intersecting planes, and sparse high-dimensional distributions.

These geometric/spectral recipes are explicit release configurations, not recovered original sample archives. The sphere and torus run on native geometry; analytic-manifold configurations must use `preprocess: none`. Gaussian mixtures and sparse clouds may use global centering/scaling plus an empirical bounded domain.

## Feature contract

Supported files contain real, finite numeric arrays:

- `.npy`: the array itself has shape `(N, n)`.
- `.npz`: the default key is `features`; set `key` to select another numeric array.

No object arrays, pickle, implicit downloads, or checkpoint discovery are used. `expected_dim` and `expected_samples` can verify the raw feature shape before optional subsampling. `sha256` can enforce a known archive checksum; when unset, the observed checksum is still recorded.

```yaml
data:
  kind: features
  path: data/resnet18.npy
  expected_samples: 60000
  expected_dim: 512
  preprocess: standardize
  sha256: null
```

`preprocess` supports `none`, `center`, and `standardize`. Standardization subtracts the column mean and divides by a **single global** standard deviation, matching the supplied reference convention. It is not coordinatewise whitening. The transform is fit once and recorded. Fixed-stream calibration configs use a seeded without-replacement subset of up to 20,000 rows; the selected row-index checksum is recorded.

Changing encoder weights, preprocessing, layer choice or text pooling changes the covariance spectrum. Matching a feature dimension alone does not reproduce an original feature substrate.

## Optional image feature export

The optional neural exporter is outside the validated core environment. It requires an appropriate PyTorch/TorchVision installation and CIFAR-10. A separate version-pinned recipe is provided in `requirements-features.txt`; no GPU feature extraction was executed for this release. Consult the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) and [TorchVision model documentation](https://docs.pytorch.org/vision/stable/models.html) for your platform.

An explicit example using a named pretrained weight enum is:

```bash
python -m pip install -r requirements-features.txt
python scripts/extract_vision.py \
  --encoder resnet18 --weights IMAGENET1K_V1 \
  --data-root data/cifar10 --split all \
  --output data/resnet18-export --download
```

This is a **new feature-extraction recipe**, not a claim that `IMAGENET1K_V1` was the manuscript checkpoint. To use a known original checkpoint, provide `--checkpoint` with a local, matching TorchVision `state_dict` and select the transform's explicit weight enum. `DEFAULT` is rejected because its meaning can change.

| Encoder argument | Feature location | Dimension |
| --- | --- | ---: |
| `resnet18` | Pooled feature before classification head | 512 |
| `vit_b_16` | Class-token representation before classification heads | 768 |
| `vgg16` | Final 4096-dimensional classifier hidden representation, in evaluation mode | 4096 |

The exporter records weight selection, transforms, library versions, split ordering, checkpoint hash where supplied, and final feature checksum. Its directory contains `features.npy`, `labels.npy`, and `metadata.json`. Change the config path to that exact feature file.

Combined CIFAR-10 train/test splits match the manuscript's 60,000-sample spectrum setting. Combining splits here is an unsupervised feature-spectrum diagnostic and must not be reused as a supervised held-out evaluation protocol.

## Optional text feature export

Prepare a local pretrained encoder/tokenizer directory and a UTF-8 JSONL file with one `text` string per row. For AG News, preserve and document the exact title/description concatenation and train/test ordering. The combined split count reported in the manuscript is 127,600.

```bash
python scripts/extract_text.py \
  --input data/ag_news.jsonl --model-dir data/bert-base-local \
  --pooling cls --max-length 128 \
  --output data/bert-export
```

`cls` and `128` are explicit example choices; original pooling and sequence-length metadata were not supplied. The exporter requires both arguments, loads only local files, rejects remote custom-code loading, and records input/model-file/feature hashes. `mean` pooling uses the attention mask over non-padding tokens, including special tokens. JSONL ordering is preserved.

External datasets and model weights are not redistributed in this repository. Their original licensing and access conditions continue to apply.
