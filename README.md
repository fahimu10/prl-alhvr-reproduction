# ALHVR Reproduction and Class-Aware Threshold Extension

## Overview

This course project reproduces **ALHVR** — *Adaptive Learning of
High-Value Regions for Semi-Supervised Medical Image Segmentation* (Tao
Lei, Ziyao Yang, Xingwu Wang, Yi Wang, Xuan Wang, Feiman Sun, Asoke K.
Nandi) — on the **ACDC** cardiac MRI dataset with 10% labeled data, and
extends it. The project:

- reproduces ALHVR, its two module ablations, and the U-Net and CPS
  baselines;
- evaluates the reproduction against the values reported in the paper;
- introduces and evaluates a class-aware confidence threshold `γ_c` as an
  alternative to ALHVR's single global threshold;
- reports additional sensitivity and capacity experiments: the threshold
  fraction `β`, the sharpening temperature `T`, network width, and a
  frozen DINOv2 encoder branch.

Both parts are complete.

## Project scope

### Part 1 — Reproduction

- **Methods:** U-Net baseline (labeled data only), CPS, full ALHVR, and
  the two Table 5 ablations (CG-CPCL only, DTCT only).
- **Setting:** ACDC, 10% labeled data, three seeds per method (1337,
  2024, 42) — 15 runs.
- **Metrics:** Dice, Jaccard, 95HD and ASD on the test set. Acceptance
  criterion: within ±5% of the paper's reported values.
- **Fidelity:** architecture, hyperparameters, training schedule and loss
  formulas are ported from the official ALHVR repository, including two
  documented quirks of the reference that are deliberately kept. The
  official repository contains no U-Net baseline or CPS training script;
  both were constructed from ALHVR's own components, as documented in
  their module docstrings.

### Part 2 — Extensions

- **Class-aware threshold `γ_c` (primary contribution).** ALHVR splits
  unlabeled pixels into regions using one threshold `γ`: the mean of the
  lowest `β` fraction of prediction confidences, pooled over all pixels.
  Because background makes up about 88% of pixels, `γ` is set almost
  entirely by background. `γ_c` instead computes one threshold per
  predicted class. Three formulations were tested: *quantile*, *offset*
  and *inverse*.
- **Context and sensitivity experiments:** `β` ∈ {0.5, 0.6, 0.7, 0.8,
  0.9}; sharpening temperature (`T` = 1.0, i.e. sharpening off, and a
  linear warm-up from 0.5 to 0.1); capacity (base width 32 instead of 16,
  4× parameters); a frozen DINOv2 encoder branch; and width 32 combined
  with DINOv2.
- **Runs:** 28. Every variant that supports a claim has three seeds; four
  exploratory variants are single-seed (two `γ_c` formulations, `T` = 1.0,
  and width 32 + DINOv2).

## Results

All values come from each run's `results.json`, written by
`src/evaluation/evaluate_acdc.py` from the checkpoint with the best
validation Dice; the two-network methods report network 1, as the
reference does. The test set is 40 volumes (20 patients × 2 cardiac
phases). Dice and Jaccard are in %. 95HD and ASD are computed with MedPy
without voxel spacing, exactly as in the official evaluation script, so
they are in voxel units. Reproduced values are mean ± standard deviation
over seeds 1337, 2024 and 42.

### Part 1: ALHVR reproduction

Values in parentheses (italic) are those reported in the ALHVR paper,
which gives single values without seeds or variance.

**Methods (paper Table 2, ACDC, 10% labeled)**

| Method | Dice | Jaccard | 95HD | ASD |
|---|---|---|---|---|
| U-Net baseline | 83.44 ± 1.72 *(81.59)* | 72.95 ± 2.19 *(70.74)* | 9.97 ± 2.70 *(8.07)* | 2.79 ± 0.56 *(2.35)* |
| CPS | 87.01 ± 0.58 *(85.34)* | 77.87 ± 0.79 *(75.50)* | 5.53 ± 0.82 *(8.78)* | 1.65 ± 0.07 *(2.37)* |
| **ALHVR** | **89.77 ± 0.33** *(90.56)* | **81.95 ± 0.52** *(83.21)* | **2.45 ± 0.11** *(2.57)* | **0.75 ± 0.13** *(0.78)* |

**Ablations (paper Table 5)**

| Variant | Dice | Jaccard | 95HD | ASD |
|---|---|---|---|---|
| CG-CPCL only | 89.11 ± 0.38 *(89.32)* | 80.90 ± 0.56 *(81.31)* | 2.91 ± 0.69 *(3.45)* | 0.90 ± 0.27 *(1.02)* |
| DTCT only | 88.47 ± 0.76 *(89.19)* | 80.06 ± 1.23 *(80.97)* | 6.69 ± 3.91 *(3.26)* | 1.82 ± 0.94 *(0.94)* |
| Full ALHVR | 89.77 ± 0.33 *(90.56)* | 81.95 ± 0.52 *(83.21)* | 2.45 ± 0.11 *(2.57)* | 0.75 ± 0.13 *(0.78)* |

- **ALHVR reproduces within ±5% of the paper on all four metrics**, using
  the three-seed mean (Dice −0.9%, Jaccard −1.5%, 95HD −4.5%, ASD −3.3%).
- Every method is within ±5% on Dice and Jaccard. The method ordering
  (U-Net < CPS < ALHVR) and the ablation ordering (each module helps on
  its own, both together help most) also reproduce.
- Boundary metrics deviate more for the other methods. DTCT-only 95HD is
  +105% from the paper's value, driven by a few cases with stray
  components. CPS has lower 95HD and ASD than published (−37% and −30%),
  plausibly because our CPS is a construction rather than a port. The
  U-Net baseline's +23.6% 95HD lies within its own seed spread. Across
  methods, the seed-to-seed standard deviation of 95HD ranges from 4.5%
  of the mean (full ALHVR) to 58% (DTCT only).

### Part 2: extensions

Every row is full ALHVR with one change. **Δ Dice** is the paired
difference in mean Dice from the ALHVR baseline (published `β` = 0.8),
computed over the same 40 test volumes after averaging each variant's
per-case values across its seeds; **p** is from the corresponding paired
test. Eleven comparisons were made, so the Bonferroni-corrected threshold
is p < 0.005; p-values in bold clear it. Rows with n = 1 are single-seed
exploratory runs (seed 1337).

| Experiment | Variant | n | Dice | Δ Dice | p |
|---|---|---|---|---|---|
| Reference | ALHVR, `β` = 0.8 | 3 | 89.77 ± 0.33 | — | — |
| Class-aware threshold `γ_c` | inverse, α = 0.5 | 3 | 90.07 ± 0.25 | +0.306 | 0.046 |
| | quantile | 1 | 86.99 | — | — |
| | offset, foreground classes only | 1 | 86.25 | — | — |
| Threshold fraction `β` | `β` = 0.5 | 3 | 90.28 ± 0.47 | +0.519 | 0.020 |
| | `β` = 0.6 | 3 | 90.24 ± 0.54 | +0.474 | 0.014 |
| | `β` = 0.7 | 3 | 90.01 ± 0.37 | +0.248 | 0.084 |
| | `β` = 0.9 | 3 | 90.55 ± 0.09 | +0.780 | **0.0001** |
| Sharpening temperature `T` | linear warm-up (0.5 → 0.1) | 3 | 90.25 ± 0.60 | +0.489 | 0.065 |
| | `T` = 1.0 (sharpening off) | 1 | 89.68 | −0.087 | 0.812 |
| Capacity and encoder | width 32 (4× parameters) | 3 | 90.62 ± 0.32 | +0.858 | **0.0005** |
| | frozen DINOv2 branch | 3 | 90.05 ± 0.45 | +0.280 | 0.397 |
| | width 32 + DINOv2 | 1 | 88.95 | — | — |

Per-class, Jaccard and boundary-metric results for every run are in that
run's `results.json`.

## Key findings

1. **Among the five `β` values tested, ALHVR's published `β` = 0.8 gave
   the lowest mean Dice.** `β` = 0.9 improved Dice by +0.78 (p = 0.0001)
   and raised every class (RV +0.99, p = 0.0013; myocardium +0.59,
   p = 0.0004; LV +0.75, p = 0.015). It also had the smallest
   seed-to-seed spread of any three-seed variant (σ = 0.09) and the
   lowest mean 95HD and ASD. However, no 95HD comparison reaches
   p < 0.05 at three seeds. This finding concerns ALHVR's published
   configuration and is not a contribution of this project.
2. **More capacity improved Dice, but no more than `β` = 0.9.** Width 32
   gave +0.86 Dice (p = 0.0005), but was statistically indistinguishable
   from `β` = 0.9 head to head (+0.078, p = 0.56). It also took roughly
   1.7× the training time and had a higher mean 95HD (3.28 vs 1.93). The
   time ratio is indicative only, because the GPU type of each run was
   not recorded. Among the tested settings, `β` = 0.9 reached the same
   accuracy at lower cost.
3. **For `γ_c`, the direction of the per-class adjustment mattered.**
   - The quantile and offset formulations derive each class's threshold
     from that class's own confidences. Foreground classes are less
     confident than background, so their thresholds fell and fewer of
     their pixels reached ALHVR's two modules. At iteration 2000 the
     quantile formulation cut RV's low-confidence coverage from 58% to
     13%. Both formulations reduced Dice by 2.8–3.5 points (one seed
     each).
   - The inverse formulation raises the threshold for under-confident
     classes. It gave +0.31 Dice over the baseline (p = 0.046, three
     seeds), which does not clear the Bonferroni threshold. RV Dice rose
     by 0.59, short of the +1.0 hypothesised in the Part 2 proposal.
   - The direction of the adjustment is clearly established (inverse vs
     quantile: +3.085 Dice, p < 0.0001). Its benefit over the global
     threshold is marginal.

Secondary results:

- **Frozen DINOv2 features** gave no overall gain (+0.28, p = 0.40). RV
  improved by 1.50 (p = 0.0053) while LV fell by 0.66 and 95HD worsened.
  The input normalisation was not matched to DINOv2's pretraining, so
  this result applies to DINOv2 as integrated here. Width 32 combined
  with DINOv2 (one seed) scored 88.95, below the baseline and below
  either change on its own.
- **Sharpening:** turning off DTCT's sharpening step (`T` = 1.0, one
  seed) changed Dice by −0.09 (p = 0.81). The linear warm-up gave +0.49
  (p = 0.065), which is not significant.
- **Boundary metrics:** no Part 2 comparison of 95HD reaches p < 0.05 at
  three seeds, so 95HD and ASD differences are directional only.

## Repository structure

```text
.
├── src/
│   ├── datasets/      ACDC loader (port of ALHVR's BaseDataSets, RandomGenerator,
│   │                  TwoStreamBatchSampler); prepare_acdc.py builds the dataset
│   ├── models/        UNet (baseline, CPS) and UNet_fea_aux (ALHVR), with the
│   │                  Part 2 width and frozen-DINOv2 options
│   ├── losses/        DiceLoss, ported from the reference
│   ├── training/      train_unet_baseline.py, train_cps.py, train_alhvr.py
│   │                  (ablations and all Part 2 variants via flags), smoke_test.py
│   ├── evaluation/    evaluate_acdc.py (test metrics -> results.json), val_2d.py,
│   │                  make_figures.py, plot_training.py, check_spurious_components.py
│   └── utils/         prototypes, ramp-up schedule, device selection,
│                      run naming, metric logging
├── slurm/             train.slurm (single job), train_array.slurm (Part 1 job array)
├── requirements.txt
└── LICENSE
```

Created locally and not tracked in Git:

- `reference/alhvr-official/` — clone of the official ALHVR repository;
  it is required for dataset preparation.
- `data/` — the prepared dataset.
- `outputs/` — one directory per run, holding `metrics.csv`,
  `config.json`, `results.json`, `log.txt` and checkpoints.
- `reports/figures/` — figures generated from `outputs/`.

## Reproducing the experiments

Run every step from the repository root. Each entry point accepts
`--help`, which lists its options.

| Step | Entry point |
|---|---|
| Prepare the dataset | `python -m src.datasets.prepare_acdc` |
| Check the pipeline | `python -m src.training.smoke_test` |
| Train | `python -m src.training.train_unet_baseline`, `train_cps`, `train_alhvr` |
| Train on TinyGPU | `sbatch.tinygpu slurm/train.slurm` (one run), `sbatch.tinygpu slurm/train_array.slurm` (Part 1 job array) |
| Evaluate | `python -m src.evaluation.evaluate_acdc` |
| Figures | `python -m src.evaluation.make_figures`, `python -m src.evaluation.plot_training` |

- PyTorch is installed separately, matched to the local CUDA version; all
  other dependencies are pinned in `requirements.txt`.
- All hyperparameters default to ALHVR's own values (batch size 16,
  30,000 iterations, learning rate 0.01, `β` = 0.8, `T` = 0.1). Do not
  change `batch_size` for reported runs.
- Boundary metrics are not bit-reproducible, even at a fixed seed,
  because a CUDA upsampling kernel is non-deterministic; compare Dice and
  Jaccard first.
- Trained checkpoints are not included in the repository.

## Dataset

- **Not included.** The ACDC data is not included in this repository
  (`data/` is gitignored). Download the official ACDC training set from
  the MICCAI 2017 challenge organisers.
- **Preparation.** `src/datasets/prepare_acdc.py` converts the raw NIfTI
  volumes to the `.h5` layout the loaders expect, using the reference
  normalisation. It copies the authors' train/validation/test split
  lists from the official ALHVR repository rather than regenerating
  them.
- **Provenance.** The dataset is generated from the official release, not
  from a third-party preprocessed mirror.
- **Splits:** 1,312 training slices, 20 validation volumes and 40 test
  volumes.
- **Labeled subset.** The 10% subset is the first 136 training slices (7
  patients). It is fixed rather than sampled, so the reported seed
  spread covers training stochasticity only.

With the raw ACDC training set in `data/acdc_raw/training` and the
official repository cloned to `reference/alhvr-official/`, run:

```bash
python -m src.datasets.prepare_acdc --raw-dir data/acdc_raw/training --out-dir data/acdc_processed/data
```

## License

This project's own work is licensed under
[CC BY-NC 4.0](LICENSE) (Attribution-NonCommercial). That covers the
`γ_c` extension, documentation, scripts and other code written for this
project.

Parts of `src/` are ported or adapted from the official ALHVR
implementation, which declared no license when it was cloned for this
project. The CC BY-NC 4.0 license does not extend to those ported
portions, which remain subject to whatever rights the ALHVR authors
hold. See [LICENSE](LICENSE). Each ported file names its upstream
source in its module docstring.

## Acknowledgements

This project builds on the ALHVR paper and its official implementation:

- Tao Lei, Ziyao Yang, Xingwu Wang, Yi Wang, Xuan Wang, Feiman Sun and
  Asoke K. Nandi. *Adaptive Learning of High-Value Regions for
  Semi-Supervised Medical Image Segmentation.*
- Official implementation:
  [github.com/ziziyao/ALHVR](https://github.com/ziziyao/ALHVR). It is the
  sole reference for every implementation detail reproduced here.

The official implementation states that its code is adapted from CCT,
MC-Net, UPCoL and SSL4MIS, and its U-Net architecture is attributed to
PyMIC. See the ALHVR repository's README for the prior work its authors
credit.
