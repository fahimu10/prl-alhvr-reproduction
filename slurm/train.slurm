#!/bin/bash -l
#
# TinyGPU batch job for ALHVR reproduction runs.
#
# Submit from the repo root on tinyx:
#   mkdir -p logs                      # Slurm will NOT create this for you
#   sbatch.tinygpu slurm/train.slurm   # note the .tinygpu wrapper
#
# Configure via --export, e.g.:
#   sbatch.tinygpu --export=ALL,TRAINER=unet,SEED=42 slurm/train.slurm
#   sbatch.tinygpu --partition=a100 --gres=gpu:a100:1 \
#       --export=ALL,TRAINER=alhvr,SEED=1337 slurm/train.slurm
#
# GPU CHOICE MATTERS FOR ALHVR: it needs ~7.6 GB at the required
# batch_size=16 and OOMs on an 8 GB card inside CG-CPCL's cosine-similarity
# step. unet/cps are comfortable on the rtx3080 default; override ALHVR to
# A100 on submission. Reducing batch_size to fit is NOT an option for
# reported runs - 16 is ALHVR's own default, confirmed in paper Sec. 4.2.
#
#SBATCH --job-name=alhvr
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

unset SLURM_EXPORT_ENV
set -euo pipefail

# ---------------------------------------------------------------- parameters
TRAINER="${TRAINER:-unet}"          # unet | cps | alhvr
SEED="${SEED:-1337}"
LABELNUM="${LABELNUM:-7}"
USE_CGCPCL="${USE_CGCPCL:-1}"       # alhvr only; 1/0 -> the Table 5 ablations
USE_DTCT="${USE_DTCT:-1}"           # alhvr only
USE_CLS_THR="${USE_CLS_THR:-0}"     # alhvr only; 1 -> Part 2's class-aware gamma_c
# PART 2: which classes get their own gamma_c. "foreground" is the variant
# in the Part 2 proposal; "all" includes background.
CLS_THR_SCOPE="${CLS_THR_SCOPE:-all}"
CLS_THR_MODE="${CLS_THR_MODE:-offset}"
# "inverse" mode only: fraction of the remaining headroom (1 - gamma) the
# hardest class's threshold moves up. 1.0 puts it at gamma=1, i.e. the whole
# class low-confidence - see generate_threshold_per_class.
CLS_THR_ALPHA="${CLS_THR_ALPHA:-0.5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Short-run override for a pre-flight check; leave unset for a real run,
# where the trainers' own defaults are the reported recipe.
[ -n "${MAXITER:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --max-iterations $MAXITER"
[ -n "${VALINT:-}" ]  && EXTRA_ARGS="$EXTRA_ARGS --val-interval $VALINT"
# beta sweep: PROPORTION=0.5 etc. Non-default values get a "-bNN" suffix
# in the run name, so a sweep run can never be mistaken for the baseline.
PROPORTION="${PROPORTION:-}"
[ -n "$PROPORTION" ] && EXTRA_ARGS="$EXTRA_ARGS --proportion $PROPORTION"
# capacity experiment: WIDTH=32 doubles every channel (4x params).
WIDTH="${WIDTH:-}"
[ -n "$WIDTH" ] && EXTRA_ARGS="$EXTRA_ARGS --width $WIDTH"
# PART 2: T warm-up. TSCHED=linear|step, optional TSTART/TRAMP.
TSCHED="${TSCHED:-}"
[ -n "$TSCHED" ] && EXTRA_ARGS="$EXTRA_ARGS --temperature-schedule $TSCHED"
# PART 2: frozen DINOv2 encoder branch. USEDINO=1 to enable.
[ -n "${USEDINO:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --use-dino $USEDINO"
[ -n "${TSTART:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --temperature-start $TSTART"
[ -n "${TRAMP:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --temperature-rampup $TRAMP"

# Repo lives in $HOME (small, backed up); data/ and outputs/ inside it
# are symlinks to $WORK, so large files never touch the $HOME quota.
# Set PROJECT if the repo is cloned elsewhere.
#
# DATA_TAR is the prepared dataset packed into one file. Build it once on
# the login node, from the repo root:
#   tar cf data/acdc_processed.tar -C data/acdc_processed data
PROJECT="${PROJECT:-$HOME/alhvr-reproduction}"
DATA_TAR="${DATA_TAR:-$PROJECT/data/acdc_processed.tar}"
RUN_ROOT="$PROJECT/outputs"

echo "=== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) ==="
echo "start:   $(date --iso-8601=seconds)"
echo "trainer: $TRAINER  seed: $SEED  labelnum: $LABELNUM"
[ "$TRAINER" = "alhvr" ] && echo "modules: cgcpcl=$USE_CGCPCL dtct=$USE_DTCT class_threshold=$USE_CLS_THR/$CLS_THR_MODE/$CLS_THR_SCOPE"
echo "TMPDIR:  $TMPDIR"

# Pre-built PyTorch 2.6 + CUDA already matched to this cluster.
# Do NOT create a conda env or pip install torch here - the module
# solves the CUDA matching that a fresh env would re-fight.
module load python/pytorch2.6py3.12

nvidia-smi

# ------------------------------------------------------- stage data to TMPDIR
# ~2100 small .h5 files touched ~480k times over 30k iterations: reading that
# from $WORK starves the GPU. Unpack once to node-local SSD instead.
echo "staging dataset to node-local SSD ..."
mkdir -p "$TMPDIR/stage"
tar xf "$DATA_TAR" -C "$TMPDIR/stage"
ROOT_PATH="$TMPDIR/stage/data"
echo "  staged: $(find "$ROOT_PATH/slices" -name '*.h5' | wc -l) slices, "\
"$(find "$ROOT_PATH" -maxdepth 1 -name '*.h5' | wc -l) volumes"

# Write results to node-local SSD too, then persist before TMPDIR is wiped.
LOCAL_OUT="$TMPDIR/outputs"
mkdir -p "$LOCAL_OUT" "$RUN_ROOT"

# --------------------------------------------------------------------- train
cd "$PROJECT"

case "$TRAINER" in
  alhvr)
    srun python -m src.training.train_alhvr \
        --root-path "$ROOT_PATH" \
        --output-dir "$LOCAL_OUT" \
        --labelnum "$LABELNUM" \
        --seed "$SEED" \
        --use-cgcpcl "$USE_CGCPCL" \
        --use-dtct "$USE_DTCT" \
        --use-class-threshold "$USE_CLS_THR" \
        --class-threshold-scope "$CLS_THR_SCOPE" \
        --class-threshold-mode "$CLS_THR_MODE" \
        --class-threshold-alpha "$CLS_THR_ALPHA" \
        $EXTRA_ARGS
    ;;
  cps)
    srun python -m src.training.train_cps \
        --root-path "$ROOT_PATH" \
        --output-dir "$LOCAL_OUT" \
        --labelnum "$LABELNUM" \
        --seed "$SEED" \
        $EXTRA_ARGS
    ;;
  unet)
    srun python -m src.training.train_unet_baseline \
        --root-path "$ROOT_PATH" \
        --output-dir "$LOCAL_OUT" \
        --labelnum "$LABELNUM" \
        --seed "$SEED" \
        $EXTRA_ARGS
    ;;
  *)
    echo "unknown TRAINER='$TRAINER' (expected unet | cps | alhvr)" >&2
    exit 1
    ;;
esac

# ------------------------------------------------------------------ evaluate
# make_figures.py reads results.json, so a run without one is silently
# missing from every figure. Failures are swallowed on purpose: under
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

# ------------------------------------------------- persist before TMPDIR dies
# $TMPDIR is deleted when the job ends. Anything not copied out is lost.
echo "persisting results to $RUN_ROOT ..."
rsync -a "$LOCAL_OUT/" "$RUN_ROOT/"

echo "end:     $(date --iso-8601=seconds)"
echo "results: $RUN_ROOT"
