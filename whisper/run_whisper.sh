#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

#MODEL_IDs=("openai/whisper-tiny.en")
#MODEL_IDs=("openai/whisper-large-v3-turbo")
#MODEL_IDs=("openai/whisper-medium.en")
MODEL_IDs=("openai/whisper-small")
#BATCH_SIZE=32
BATCH_SIZE=32
NUM_BEAMS=1
MAX_NEW_TOKENS=448
DEVICE="cuda:0"
#DEVICE="cpu"
#MODEL_TYPE="onnx_audio"
MODEL_TYPE="hf_audio"
MODEL_PATH="/sunghcho_data/onnx_models/whisper-tiny-en/onnx/cuda/cuda-fp16/"
MODEL_PATH="/sunghcho_data/onnx_models/whisper-large-v3-turbo/onnx/cuda/cuda-olive-1/"
MODEL_PATH="/sunghcho_data/onnx_models/whisper-small/onnx/cpu_and_mobile/cpu-fp32/"
#MODEL_PATH="/sunghcho_data/onnx_models/whisper-small/onnx/cuda/cuda-fp16/"
#MODEL_PATH="/home/jiafa/accuracy/onnxruntime/onnxruntime/python/tools/transformers/cache_models/models--openai--whisper-tiny.en/snapshots/87c7102498dcde7456f24cfd30239ca606ed9063"
#MODEL_PATH="/home/jiafa/accuracy/onnxruntime/onnxruntime/python/tools/transformers/whisper-turbo/"
#MODEL_PATH="/sunghcho_data/onnx_models/whisper-medium-en/onnx/cuda/cuda-fp32-no-opt/"
#MODEL_PATH="/sunghcho_data/onnx_models/whisper-medium-en/onnx/cuda/cuda-fp16/"
#MODEL_PATH="/sunghcho_data/onnx_models/whisper-large-v3-turbo/onnx/cuda/cuda-fp16/"
#MODEL_PATH="/home/jiafa/accuracy/onnxruntime/onnxruntime/python/tools/transformers/cache_models/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/"

num_models=${#MODEL_IDs[@]}
default_user_prompt="Transcribe the audio clip into text."

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}" \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}" \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="Transcribe the audio clip to English text." \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"
        

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}" \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}" \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}" \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}" \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE} \
        --batch_size=${BATCH_SIZE} \
        --num_beams=${NUM_BEAMS} \
        --max_eval_samples=-1 \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        --user_prompt="${default_user_prompt}" \
        --model_type="${MODEL_TYPE}" \
        --model_path="${MODEL_PATH}"

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
