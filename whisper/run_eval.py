import argparse
from email.mime import audio
import itertools
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
import onnxruntime_genai as og
# og.set_log_options(enabled=True, model_input_values=True, model_output_values=False)
import io
import soundfile as sf
            

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')

import hashlib
import numpy as np

def deterministic_sample_fn(sampling_rate=1.0/32, seed=42, key="id"):
    """
    Returns a filter function that keeps ~sampling_rate of the dataset
    consistently based on a hash of a given key field (e.g. 'id').
    
    Works with datasets.IterableDataset.filter()
    """
    def filter_fn(example):
        id_str = str(example[key]) + str(seed)
        hash_val = int(hashlib.sha1(id_str.encode("utf-8")).hexdigest(), 16)
        rand_val = (hash_val % 10_000_000) / 10_000_000
        return rand_val < sampling_rate

    return filter_fn

class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


def main(args):
    if args.model_type == "hf_audio":
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id, trust_remote_code=True).to(args.device)
        model.config.forced_decoder_ids = None
        model.eval()
        processor = WhisperProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        if args.model_id.endswith(".en"):
            prompt_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        else:
            prompt_ids = None
        print("Prompt token IDs:", prompt_ids)
    elif args.model_type == "onnx_audio":
        config = og.Config(args.model_path)
        config.clear_providers()
        # args.execution_provider != "cpu":
        if args.device.startswith("cuda"):
            config.append_provider("cuda")
        model = og.Model(config)
        processor = model.create_multimodal_processor()
        tokenizer = og.Tokenizer(model)

    gen_kwargs = {"num_beams": args.num_beams} # "max_new_tokens": args.max_new_tokens
    gen_kwargs.update({
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95
    })
    if "en" not in args.model_id:
        gen_kwargs["language"] = "en"

    '''
    stop_tokens = [prompt_suffix, processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(model.device)
    '''
    def benchmark_hf(batch, min_new_tokens=None):
        # Load audio inputs
        audios_array = [audio["array"] for audio in batch["audio"]]
        audios_sampling_rate = batch["audio"][0]["sampling_rate"]
        minibatch_size = len(audios_array)
        '''
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=args.num_beams * minibatch_size)]
        )
        '''
        # START TIMING
        start_time = time.time()

        with torch.autocast(model.device.type, enabled=True):
            inputs = processor(audios_array, sampling_rate=audios_sampling_rate, return_tensors="pt", padding="max_length")
            input_features = inputs.input_features.to(args.device)            
            
            # Model Inference
            pred_ids  = model.generate(
                input_features,
                **gen_kwargs,         
            )
            # print(f"pred_ids: {pred_ids}")
        
        pred_text = [
            processor.decode(_pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids in pred_ids
        ]
        # END TIMING
        runtime = time.time() - start_time
        # print(f"hf pred_text: {pred_text}")

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch
    
    def benchmark_ort(batch, min_new_tokens=None):
        audio_bytes = []
        for audio in batch["audio"]:
            buffer = io.BytesIO()
            sf.write(buffer, audio["array"], samplerate=audio["sampling_rate"], format='WAV')
            audio_bytes.append(buffer.getvalue())
        minibatch_size = len(audio_bytes)
        audios = og.Audios.open_bytes(*audio_bytes)
        if args.model_id.endswith(".en"):
            decoder_prompt_tokens = ["<|startoftranscript|>", "<|notimestamps|>"]
        else:
            decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
        prompts = ["".join(decoder_prompt_tokens)] * minibatch_size
        inputs = processor(prompts, audios=audios)

        audios_array = [audio["array"] for audio in batch["audio"]]
        audios_sampling_rate = batch["audio"][0]["sampling_rate"]
        minibatch_size = len(audios_array)

        #torch_inputs = torch_processor(audios_array, sampling_rate=audios_sampling_rate, return_tensors="pt")
        #torch_input_array = np.array(torch_inputs.input_features)
        #inputs['audio_features'] = torch_input_array

        params = og.GeneratorParams(model)
        num_return_sequences = 1
        if args.num_beams > 1:
            params.set_search_options(do_sample=False, max_length=448, min_length=0, batch_size=minibatch_size, num_beams=args.num_beams, num_return_sequences=num_return_sequences)  
        else:
            top_k = 50
            # top_p=0.95 # 1.0
            # do_sample = top_k > 1 or (top_p != 1.0 and top_p > 0.0) # False
            top_p = 1.0
            # For debug only
            do_sample = False 
            params.set_search_options(do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=1.0, max_length=448, min_length=0, batch_size=minibatch_size)

        # START TIMING
        start_time = time.time()

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        while not generator.is_done():
            generator.generate_next_token()
        
        pred_text = []
        for i in range(minibatch_size * num_return_sequences):
            tokens = generator.get_sequence(i)
            transcription = processor.decode(tokens)
            pred_text.append(transcription.strip())
        # END pred_text
        runtime = time.time() - start_time
        # print(f"ort pred_text: {pred_text}")

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        #print(f"batch_references={batch['references']}", flush=True)
        #print(f"batch_predictions={batch['predictions']}", flush=True)
        return batch
    
    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)
        #for data in dataset:
        #    sf.write("/home/jiafa/accuracy/open_asr_leaderboard/whisper/data/"+data['audio']['path'], data['audio']['array'], data['audio']['sampling_rate'])
        #input("stop here")

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        
        # warmup_dataset = warmup_dataset.filter(lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] < 30)
        # sample_fn = deterministic_sample_fn(sampling_rate=1.0/32, key="id")
        # warmup_dataset = warmup_dataset.filter(sample_fn)
        
        #for data in warmup_dataset:
        #    sf.write("/home/jiafa/accuracy/open_asr_leaderboard/whisper/data/"+data['audio']['path'], data['audio']['array'], data['audio']['sampling_rate'])

        if args.model_type == "hf_audio":           
            warmup_dataset = iter(warmup_dataset.map(benchmark_hf, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))
        elif args.model_type == "onnx_audio":
            torch_processor = WhisperProcessor.from_pretrained(args.model_id, trust_remote_code=True)        
            warmup_dataset = iter(warmup_dataset.map(benchmark_ort, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue
    
    dataset = data_utils.load_data(args)
    

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))            
            
    dataset = data_utils.prepare_data(dataset)
    #sample_fn = deterministic_sample_fn(sampling_rate=1.0/32, key="id")
    #dataset = dataset.filter(sample_fn)

    # dataset = dataset.filter(lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] < 30)
    #for data in dataset:
    #    sf.write("/home/jiafa/accuracy/open_asr_leaderboard/whisper/data/"+data['audio']['path'], data['audio']['array'], data['audio']['sampling_rate'])  
    # dataset = dataset.take(1)
    if args.model_type == "hf_audio":
        dataset = dataset.map(
            benchmark_hf, batch_size=args.batch_size, batched=True,
        )
    elif args.model_type == "onnx_audio":
        # print("Using ONNX Runtime for evaluation...", flush=True)
        dataset = dataset.map(
            benchmark_ort, batch_size=args.batch_size, batched=True,
        )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id + "_" + args.model_type,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default="Transcribe the audio clip into text.",
        help="User prompt string.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Model type.",
        help="hf_audio, onnx_audio.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the model.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
