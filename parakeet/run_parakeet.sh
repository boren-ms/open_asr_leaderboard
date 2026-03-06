#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.0"

MODEL_ID="nvidia/parakeet-tdt-0.6b-v3"
MODEL_PATH="/datadisks/disk1/jiafa/accuracy/onnxruntime-genai/tools/parakeet_export/onnx_models_cpu_int4"
BATCH_SIZE=1
# DEVICE="cuda"
DEVICE="cpu"

DATASETS=(
    "librispeech:test.clean"
    # "librispeech:test.other"
    # "voxpopuli:test"
    # "ami:test"
    # "earnings22:test"
    # "gigaspeech:test"
    # "spgispeech:test"
    # "tedlium:test"
)

for ds_split in "${DATASETS[@]}"; do
    IFS=":" read -r DATASET SPLIT <<< "$ds_split"

    echo "============================================================"
    echo "Evaluating: ${DATASET} / ${SPLIT}"
    echo "============================================================"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --model_path="${MODEL_PATH}" \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="${DATASET}" \
        --split="${SPLIT}" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=20
        # --max_eval_samples=-1

done

# Evaluate results
RUNDIR=$(pwd)
cd ../normalizer && \
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
cd "$RUNDIR"
