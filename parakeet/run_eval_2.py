"""
Parakeet TDT streaming — matches NVIDIA's speech_to_text_streaming_infer_rnnt.py.

Algorithm (from NVIDIA):
  - Buffer = [left_context | chunk | right_context]
  - Encoder runs on full buffer each iteration
  - Left context encoder frames are STRIPPED — only chunk frames are decoded
  - Decoder LSTM state carries forward between chunks
  - Each token is decoded exactly once

Recommended configs (from NVIDIA):
  - Buffered (offline-quality): left=10, chunk=10, right=5
  - Streaming (4s latency):    left=10, chunk=2,  right=2

Uses kaldi-native-fbank for mel (same as sherpa-onnx).
"""
import numpy as np
import onnxruntime as ort
import soundfile as sf
import kaldi_native_fbank as knf
import time
import sys
import os

SAMPLE_RATE = 16000
MODEL_DIR = os.path.expanduser("~/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")

DURATIONS = [0, 1, 2, 3, 4]
NUM_DURATIONS = 5
VOCAB_SIZE = 8193
BLANK_ID = 8192

# ── Context config (matches NVIDIA's recommended 10-2-2) ──
LEFT_CONTEXT_SEC = 9
CHUNK_SEC = 0.8
RIGHT_CONTEXT_SEC = 1.6

# Derived constants
ENCODER_SUBSAMPLING = 8
FRAME_SHIFT_SAMPLES = 160   # 10ms at 16kHz
ENCODER_FRAME_SAMPLES = FRAME_SHIFT_SAMPLES * ENCODER_SUBSAMPLING  # 1280

# Align to encoder frame boundaries
LEFT_SAMPLES = int(LEFT_CONTEXT_SEC * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES
CHUNK_SAMPLES = int(CHUNK_SEC * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES
RIGHT_SAMPLES = int(RIGHT_CONTEXT_SEC * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES

LEFT_ENC_FRAMES = LEFT_SAMPLES // ENCODER_FRAME_SAMPLES
CHUNK_ENC_FRAMES = CHUNK_SAMPLES // ENCODER_FRAME_SAMPLES
RIGHT_ENC_FRAMES = RIGHT_SAMPLES // ENCODER_FRAME_SAMPLES


def load_vocab():
    vocab = {}
    with open(os.path.join(MODEL_DIR, "tokens.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[int(parts[1])] = parts[0]
    return vocab


def tokens_to_text(tokens, vocab):
    return "".join(vocab.get(t, "").replace("\u2581", " ") for t in tokens)


def compute_mel_raw(audio_np):
    """Compute mel using kaldi-native-fbank WITHOUT normalization.
    Returns [T, 128] numpy float32."""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hanning"
    opts.mel_opts.num_bins = 128
    opts.mel_opts.low_freq = 0
    opts.mel_opts.is_librosa = True
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, audio_np.tolist())
    fbank.input_finished()
    num_frames = fbank.num_frames_ready
    features = np.zeros((num_frames, 128), dtype=np.float32)
    for i in range(num_frames):
        features[i] = fbank.get_frame(i)
    return features  # [T, 128] unnormalized


def normalize_features(features):
    """Per-feature normalize [T, 128] -> [128, T].
    Uses Bessel's correction (N-1) to match NeMo's per_feature normalization."""
    n = features.shape[0]
    mean = features.mean(axis=0)
    if n > 1:
        var = np.sum((features - mean) ** 2, axis=0) / (n - 1)
    else:
        var = np.zeros_like(mean)
    std = np.sqrt(var) + 1e-5
    features = (features - mean) / std
    return features.T.astype(np.float32)  # [128, T]


def compute_mel_knf(audio_np):
    """Backward compatible: compute + normalize."""
    return normalize_features(compute_mel_raw(audio_np))


def tdt_decode_chunk(enc_session, dec_session, jnt_session, encoded,
                     decode_start, decode_end, dec_out, h, c):
    """TDT greedy decode on encoder frames [decode_start, decode_end).
    Decoder state (dec_out, h, c) carries forward."""
    tokens = []
    tokens_this_frame = 0
    t = decode_start
    while t < decode_end:
        enc_t = encoded[:, :, t:t+1]
        j = jnt_session.run(None, {"encoder_outputs": enc_t, "decoder_outputs": dec_out})
        logits = j[0].squeeze()
        tok = int(np.argmax(logits[:VOCAB_SIZE]))
        skip = int(np.argmax(logits[VOCAB_SIZE:VOCAB_SIZE + NUM_DURATIONS]))
        if tok != BLANK_ID:
            tokens.append(tok)
            tokens_this_frame += 1
            d = dec_session.run(None, {
                "targets": np.array([[tok]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "states.1": h, "onnx::Slice_3": c
            })
            dec_out = d[0][:, :, -1:]
            h, c = d[2], d[3]
        if skip > 0: tokens_this_frame = 0
        if tokens_this_frame >= 5: tokens_this_frame = 0; skip = 1
        if tok == BLANK_ID and skip == 0: tokens_this_frame = 0; skip = 1
        t += skip
    return tokens, dec_out, h, c


if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mars_16k.wav")
    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    duration = len(audio) / SAMPLE_RATE

    vocab = load_vocab()
    enc = ort.InferenceSession(os.path.join(MODEL_DIR, "encoder.int8.onnx"))
    dec = ort.InferenceSession(os.path.join(MODEL_DIR, "decoder.int8.onnx"))
    jnt = ort.InferenceSession(os.path.join(MODEL_DIR, "joiner.int8.onnx"))

    latency_sec = (CHUNK_SAMPLES + RIGHT_SAMPLES) / SAMPLE_RATE
    print(f"Audio: {duration:.1f}s")
    print(f"Config: left={LEFT_SAMPLES/SAMPLE_RATE:.1f}s, "
          f"chunk={CHUNK_SAMPLES/SAMPLE_RATE:.1f}s, "
          f"right={RIGHT_SAMPLES/SAMPLE_RATE:.1f}s")
    print(f"Theoretical latency: {latency_sec:.1f}s")
    print("=" * 70)

    t0 = time.time()

    # ── Initialize decoder state (persists across all chunks) ──
    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)
    d = dec.run(None, {
        "targets": np.array([[BLANK_ID]], dtype=np.int32),
        "target_length": np.array([1], dtype=np.int32),
        "states.1": h, "onnx::Slice_3": c
    })
    dec_out = d[0][:, :, -1:]
    h, c = d[2], d[3]

    all_tokens = []

    # ── Streaming loop: matches NVIDIA's iteration exactly ──
    chunk_start = 0

    while chunk_start < len(audio):
        chunk_end = min(chunk_start + CHUNK_SAMPLES, len(audio))
        is_last = chunk_end >= len(audio)

        # ── Build encoder window: [left_context | chunk | right_context] ──
        win_left = max(0, chunk_start - LEFT_SAMPLES)
        win_right = min(chunk_end + RIGHT_SAMPLES, len(audio))

        window_audio = audio[win_left:win_right]

        # ── Compute raw mel on the window ──
        window_raw_mel = compute_mel_raw(window_audio)  # [T, 128]

        # ── Per-window normalization (matches NeMo per_feature with Bessel's correction) ──
        mel_norm = normalize_features(window_raw_mel)  # [128, T]

        # ── Run encoder on normalized window ──
        mel_in = mel_norm[np.newaxis, :, :]
        length = np.array([mel_norm.shape[1]], dtype=np.int64)
        e = enc.run(None, {"audio_signal": mel_in, "length": length})
        encoded = e[0]   # [1, 1024, T_enc]
        enc_total = int(e[1][0])

        # ── Strip left context: calculate encoder frame where chunk starts ──
        left_ctx_samples = chunk_start - win_left
        left_ctx_mel = left_ctx_samples // FRAME_SHIFT_SAMPLES
        left_enc = left_ctx_mel // ENCODER_SUBSAMPLING

        # ── Decode only chunk frames (skip left context + right context) ──
        if is_last:
            # Last chunk: decode everything after left context
            decode_start = left_enc
            decode_end = enc_total
        else:
            decode_start = left_enc
            chunk_samples_actual = chunk_end - chunk_start
            chunk_mel = chunk_samples_actual // FRAME_SHIFT_SAMPLES
            chunk_enc = chunk_mel // ENCODER_SUBSAMPLING
            decode_end = min(left_enc + chunk_enc, enc_total)

        if decode_end <= decode_start:
            chunk_start = chunk_end
            continue

        # ── TDT decode chunk frames, carrying state forward ──
        chunk_tokens, dec_out, h, c = tdt_decode_chunk(
            enc, dec, jnt, encoded, decode_start, decode_end, dec_out, h, c
        )

        all_tokens.extend(chunk_tokens)
        new_text = tokens_to_text(chunk_tokens, vocab)
        if new_text:
            print(new_text, end="", flush=True)

        # ── Advance by chunk ──
        chunk_start = chunk_end

    print()

    dt = time.time() - t0
    rtfx = duration / dt
    print(f"\nStreaming time: {dt:.2f}s  RTFx: {rtfx:.2f}x")
    print(f"\n{'='*70}")
    print(f"STREAMING TRANSCRIPT:")
    print(f"{'='*70}")
    print(tokens_to_text(all_tokens, vocab))

    # ── Batch decode for comparison ──
    print(f"\n{'='*70}")
    print(f"BATCH TRANSCRIPT:")
    print(f"{'='*70}")
    mel_full = compute_mel_knf(audio)
    mel_in = mel_full[np.newaxis, :, :]
    length = np.array([mel_full.shape[1]], dtype=np.int64)
    e = enc.run(None, {"audio_signal": mel_in, "length": length})
    encoded_full = e[0]
    enc_len_full = int(e[1][0])

    h_b = np.zeros((2, 1, 640), dtype=np.float32)
    c_b = np.zeros((2, 1, 640), dtype=np.float32)
    d = dec.run(None, {
        "targets": np.array([[BLANK_ID]], dtype=np.int32),
        "target_length": np.array([1], dtype=np.int32),
        "states.1": h_b, "onnx::Slice_3": c_b
    })
    dec_out_b = d[0][:, :, -1:]
    h_b, c_b = d[2], d[3]
    batch_tokens, _, _, _ = tdt_decode_chunk(
        enc, dec, jnt, encoded_full, 0, enc_len_full, dec_out_b, h_b, c_b
    )
    batch_text = tokens_to_text(batch_tokens, vocab)
    print(batch_text)

    # ── WER: streaming vs batch ──
    streaming_text = tokens_to_text(all_tokens, vocab)
    def compute_wer(ref, hyp):
        r = ref.strip().lower().split()
        h = hyp.strip().lower().split()
        n, m = len(r), len(h)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                dp[i][j] = dp[i-1][j-1] if r[i-1] == h[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[n][m], n

    edits, ref_len = compute_wer(batch_text, streaming_text)
    wer = edits / max(ref_len, 1) * 100
    print(f"\n{'='*70}")
    print(f"WER (streaming vs batch): {wer:.1f}% ({edits} edits / {ref_len} ref words)")