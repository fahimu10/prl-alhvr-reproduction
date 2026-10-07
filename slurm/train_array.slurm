#!/bin/bash -l
#
# One run per array task. INDEPENDENT single-GPU jobs, not distributed
# training - do not request --gres=gpu:4, which this code cannot use.
#
# Submit from the repo root on tinyx:
#   mkdir -p logs
#   sbatch.tinygpu slurm/train_array.slurm
#
# RTX 3080 by default: ALHVR fits at batch_size=16 on 10 GB, and there are
# 56 of them against 32 A100s. A100 is 1.43x faster (151 vs 217 min for full
# ALHVR):
#   sbatch.tinygpu --partition=a100 --gres=gpu:a100:1 slurm/train_array.slurm
#
# %3 throttles to 3 concurrent tasks; raise it only if the queue is empty.
#
#SBATCH --job-name=alhvr-array
#SBATCH --array=0-8%3
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

unset SLURM_EXPORT_ENV
set -euo pipefail

# ---------------------------------------------------------------- run table
# One line per run: TRAINER USE_CGCPCL USE_DTCT SEED
# (the module flags are ignored for trainers other than alhvr)
#
# 0-8 are the ALHVR family, which needs a GPU with >8 GB. CPS (9-11) and
# the baseline (12-14) fit on an 8 GB GPU and are excluded by the default
# --array range.
#
#   --array=0-8%3    ALHVR family only (the default below)
#   --array=0-14%3   every Part 1 run
#   --array=9-14%3   the two baselines only
#
# Each task trains AND evaluates, so every run directory ends up with the
# results.json make_figures.py needs.
CONFIGS=(
  "alhvr 1 1 1337"   # 0  full ALHVR
  "alhvr 1 1 2024"   # 1
  "alhvr 1 1 42"     # 2
  "alhvr 1 0 1337"   # 3  CG-CPCL only  (Table 5)
  "alhvr 1 0 2024"   # 4
  "alhvr 1 0 42"     # 5
  "alhvr 0 1 1337"   # 6  DTCT only     (Table 5)
  "alhvr 0 1 2024"   # 7
  "alhvr 0 1 42"     # 8
  "cps   1 1 1337"   # 9  CPS baseline
  "cps   1 1 2024"   # 10
  "cps   1 1 42"     # 11
  "unet  1 1 1337"   # 12 U-Net baseline (module flags ignored)
  "unet  1 1 2024"   # 13
  "unet  1 1 42"     # 14
)

read -r TRAINER USE_CGCPCL USE_DTCT SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

LABELNUM="${LABELNUM:-7}"
# Part 2's class-aware gamma_c. An env var rather than a run-table column, so
# any existing index can be re-run with it:
#   sbatch.tinygpu --array=0-2%3 --export=ALL,USE_CLS_THR=1 slurm/train_array.slurm
# gives the three full-ALHVR seeds with gamma_c on. Ignored for unet/cps.
USE_CLS_THR="${USE_CLS_THR:-0}"
# PART 2: which classes get their own gamma_c. "foreground" is the variant
# in the Part 2 proposal; "all" includes background.
CLS_THR_SCOPE="${CLS_THR_SCOPE:-all}"
CLS_THR_MODE="${CLS_THR_MODE:-offset}"
CLS_THR_ALPHA="${CLS_THR_ALPHA:-0.5}"

# Short-run override for a pre-flight check, e.g.
#   sbatch.tinygpu --array=0 --time=00:30:00 --export=ALL,MAXITER=300,VALINT=100 ...
# Leave unset for a real run - the trainers' own defaults are the recipe.
EXTRA=""
[ -n "${MAXITER:-}" ] && EXTRA="$EXTRA --max-iterations $MAXITER"
[ -n "${VALINT:-}" ]  && EXTRA="$EXTRA --val-interval $VALINT"
# beta sweep: PROPORTION=0.5 etc. Non-default values get a "-bNN" suffix
# in the run name, so a sweep run can never be mistaken for the baseline.
PROPORTION="${PROPORTION:-}"
[ -n "$PROPORTION" ] && EXTRA="$EXTRA --proportion $PROPORTION"
# capacity experiment: WIDTH=32 doubles every channel (4x params).
WIDTH="${WIDTH:-}"
[ -n "$WIDTH" ] && EXTRA="$EXTRA --width $WIDTH"
# PART 2: T warm-up. TSCHED=linear|step, optional TSTART/TRAMP.
TSCHED="${TSCHED:-}"
[ -n "$TSCHED" ] && EXTRA="$EXTRA --temperature-schedule $TSCHED"
# PART 2: frozen DINOv2 encoder branch. USEDINO=1 to enable.
[ -n "${USEDINO:-}" ] && EXTRA="$EXTRA --use-dino $USEDINO"
[ -n "${TSTART:-}" ] && EXTRA="$EXTRA --temperature-start $TSTART"
[ -n "${TRAMP:-}" ] && EXTRA="$EXTRA --temperature-rampup $TRAMP"
# Repo lives in $HOME (small, backed up); data/ and outputs/ inside it
# are symlinks to $WORK, so large files never touch the $HOME quota.
# Set PROJECT if the repo is cloned elsewhere. DATA_TAR is built as
# described in slurm/train.slurm.
PROJECT="${PROJECT:-$HOME/alhvr-reproduction}"
DATA_TAR="${DATA_TAR:-$PROJECT/data/acdc_processed.tar}"
RUN_ROOT="$PROJECT/outputs"

echo "=== array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) ==="
echo "start:   $(date --iso-8601=seconds)"
echo "trainer: $TRAINER  seed: $SEED  labelnum: $LABELNUM"
[ "$TRAINER" = "alhvr" ] && echo "modules: cgcpcl=$USE_CGCPCL dtct=$USE_DTCT class_threshold=$USE_CLS_THR/$CLS_THR_MODE/$CLS_THR_SCOPE"

# Pre-built PyTorch + CUDA matched to this cluster. Do NOT pip install
# torch here - the module already solves the CUDA matching.
module load python/pytorch2.6py3.12
nvidia-smi

# Stage to node-local $TMPDIR: 2100 small files read from $WORK by several
# concurrent jobs is far worse than one copy per task.
echo "staging dataset to node-local SSD ..."
mkdir -p "$TMPDIR/stage"
tar xf "$DATA_TAR" -C "$TMPDIR/stage"
ROOT_PATH="$TMPDIR/stage/data"
echo "  staged: $(find "$ROOT_PATH/slices" -name '*.h5' | wc -l) slices, "\
"$(find "$ROOT_PATH" -maxdepth 1 -name '*.h5' | wc -l) volumes"

LOCAL_OUT="$TMPDIR/outputs"
mkdir -p "$LOCAL_OUT" "$RUN_ROOT"

cd "$PROJECT"

case "$TRAINER" in
  alhvr)
    srun python -m src.training.train_alhvr $EXTRA \
        --root-path "$ROOT_PATH" --output-dir "$LOCAL_OUT" \
        --labelnum "$LABELNUM" --seed "$SEED" \
        --use-cgcpcl "$USE_CGCPCL" --use-dtct "$USE_DTCT" \
        --use-class-threshold "$USE_CLS_THR" \
        --class-threshold-scope "$CLS_THR_SCOPE" \
        --class-threshold-mode "$CLS_THR_MODE" \
        --class-threshold-alpha "$CLS_THR_ALPHA"
    ;;
  cps)
    srun python -m src.training.train_cps $EXTRA \
        --root-path "$ROOT_PATH" --output-dir "$LOCAL_OUT" \
        --labelnum "$LABELNUM" --seed "$SEED"
    ;;
  unet)
    srun python -m src.training.train_unet_baseline $EXTRA \
        --root-path "$ROOT_PATH" --output-dir "$LOCAL_OUT" \
        --labelnum "$LABELNUM" --seed "$SEED"
    ;;
  *)
    echo "unknown TRAINER='$TRAINER'" >&2; exit 1 ;;
esac

# Evaluate in the same job - make_figures.py needs results.json and the
# dataset is already staged. Failures are swallowed on purpose: under
# `set -e` a broken post-processing step would abort before the rsync and
# discard hours of training.
RUN_DIR="$(find "$LOCAL_OUT" -mindepth 1 -maxdepth 1 -type d | head -1)"
case "$TRAINER" in
  alhvr) EVAL_MODEL=unet_fea_aux; CKPT="$RUN_DIR/best_model1.pth" ;;
  cps)   EVAL_MODEL=unet;         CKPT="$RUN_DIR/best_model1.pth" ;;
  unet)  EVAL_MODEL=unet;         CKPT="$RUN_DIR/best_model.pth"  ;;
esac
if [ -f "$CKPT" ]; then
  echo "evaluating $CKPT ..."
  srun python -m src.evaluation.evaluate_acdc \
      --root-path "$ROOT_PATH" --checkpoint "$CKPT" --model "$EVAL_MODEL" \
      --results-json "$RUN_DIR/results.json" \
    || echo "WARNING: evaluation failed; training output is still persisted below"
else
  echo "WARNING: no checkpoint at $CKPT - skipping evaluation"
fi

# $TMPDIR is wiped when the task ends - persist first. Run dirs are
# timestamped and carry method+seed, so concurrent tasks cannot collide.
echo "persisting results to $RUN_ROOT ..."
rsync -a "$LOCAL_OUT/" "$RUN_ROOT/"

echo "end:     $(date --iso-8601=seconds)"
