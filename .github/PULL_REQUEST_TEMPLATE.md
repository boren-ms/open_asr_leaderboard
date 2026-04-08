## Description
Please include a summary of your pull request and how to run/use it.

## Type of change
- [ ] New model
- [ ] New dataset
- [ ] Bug fix
- [ ] New feature
- [ ] Other

## New Model Checklist
- [ ] `run_eval.py` script like [this](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_eval.py). 
    - [ ] Supports batch processing.
    - [ ] Uses torch.compile and/or relevant optimization for inference (including warmup).
- [ ] Bash script like [this](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh).
    - [ ] Loops over all the [Open ASR Leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard) subsets.
    - [ ] Tested on A100-SXM4-80GB GPU with maximum possible batch size (*if you don't have access to such a GPU, let us know so we can run it*).
- [ ] (*Open models*) Results are reported on the HF Hub by adding a `.eval_results/open_asr_leaderboard.yaml` file like [this](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/blob/main/.eval_results/open_asr_leaderboard.yaml) in the model card.
- [ ] (*Closed models*) Report your results in the PR text.
- [ ] (*Models NOT in the Transformers library*) Besides the main requirements [here](https://github.com/huggingface/open_asr_leaderboard/blob/main/requirements/requirements.txt), a requirements file for the necessary dependencies as seen [here](https://github.com/huggingface/open_asr_leaderboard/tree/main/requirements).

## New Dataset Checklist
- [ ] The dataset is hosted on the HF Hub.
- [ ] Create a new Bash script with one of the model suites. For example, adapting the [Whisper](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh) or [Voxtral](https://github.com/huggingface/open_asr_leaderboard/blob/main/voxtral/run_voxtral.sh) script to run the eval on that new dataset (adding a call like [this](https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh#L14-L21)).

## Related issues
Closes #