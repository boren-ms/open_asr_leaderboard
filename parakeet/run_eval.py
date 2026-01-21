#!/usr/bin/env python3
"""
Evaluate Parakeet-TDT ASR on Open ASR Leaderboard datasets using onnxruntime-genai.

The model is loaded as nemotron_asr type via og.Model. Audio is preprocessed
with NeMo's FilterbankFeatures to produce mel spectrograms, then fed to the
ORT GenAI generator.

Usage:
    python run_eval.py \
        --model_id nvidia/parakeet-tdt-0.6b-v3 \
        --model_path /path/to/onnx_models \
        --dataset librispeech --split test.clean \
        --device cuda:0
"""

import argparse
import io
import os
import time

import evaluate
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

import onnxruntime_genai as og
from normalizer import data_utils

wer_metric = evaluate.load("wer")


def create_mel_preprocessor():
    """Create NeMo mel feature extractor matching Parakeet config.
    Always placed on CUDA to avoid CUDA arch detection issues on CPU-only runs.
    """
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

    preprocessor = FilterbankFeatures(
        sample_rate=16000,
        n_window_size=400,   # 25ms
        n_window_stride=160, # 10ms
        n_fft=512,
        nfilt=128,
        dither=0.0,
        pad_to=0,
        normalize="per_feature",
    )
    if torch.cuda.is_available():
        preprocessor = preprocessor.cuda()
    return preprocessor


def audio_to_mel(preprocessor, audio_array, sr=16000):
    """Convert raw audio waveform to mel features [1, 128, T]."""
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
    length = torch.tensor([waveform.shape[1]])
    # FilterbankFeatures has no learnable parameters; check buffers instead
    bufs = list(preprocessor.buffers())
    device = bufs[0].device if bufs else torch.device("cpu")
    features, feat_length = preprocessor(waveform.to(device), length.to(device))
    return features.cpu().numpy().astype(np.float32)


def main(args):
    # Load model
    if args.device.startswith("cuda"):
        config = og.Config(args.model_path)
        config.clear_providers()
        config.append_provider("cuda")
        model = og.Model(config)
    else:
        model = og.Model(args.model_path)

    tokenizer = og.Tokenizer(model)
    preprocessor = create_mel_preprocessor()

    # Read blank_id from genai_config
    import json
    genai_cfg_path = os.path.join(args.model_path, "genai_config.json")
    with open(genai_cfg_path) as f:
        genai_cfg = json.load(f)
    blank_id = genai_cfg["model"]["eos_token_id"]

    def benchmark(batch, min_new_tokens=None):
        minibatch_size = len(batch["audio"])

        start_time = time.time()

        pred_text = []
        for audio in batch["audio"]:
            audio_array = np.float32(audio["array"])

            # Compute mel features
            mel_np = audio_to_mel(preprocessor, audio_array, audio["sampling_rate"])

            # Run ORT GenAI
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=512, batch_size=1)
            generator = og.Generator(model, params)

            inputs = og.NamedTensors()
            inputs["audio_signal"] = mel_np
            inputs["input_ids"] = np.array([[0]], dtype=np.int32)
            generator.set_inputs(inputs)

            step = 0
            while not generator.is_done():
                generator.generate_next_token()
                step += 1
                if step > 500:
                    break

            tok_list = list(generator.get_sequence(0))
            # Filter out BOS (first token) and blank tokens
            text_ids = np.array([t for t in tok_list[1:] if t != blank_id], dtype=np.int32)
            if len(text_ids) > 0:
                text = tokenizer.decode(text_ids)
            else:
                text = ""
            pred_text.append(text.strip())

            # Cleanup to avoid OGA leak warnings
            del generator, params, inputs

        runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    # Warmup
    if args.warmup_steps is not None and args.warmup_steps > 0:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))

        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Eval
    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True)

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)

    # Cleanup
    del tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier (e.g. nvidia/parakeet-tdt-0.6b-v3)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted", help="Dataset path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (samples processed per map call)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Number of warmup steps")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming")
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
