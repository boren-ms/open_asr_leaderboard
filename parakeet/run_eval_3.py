"""
Parakeet TDT 0.6B v3 ONNX FP32 — Streaming + Batch inference.

Matches NVIDIA NeMo streaming algorithm exactly:
  - Buffer = [left_context | chunk | right_context]
  - Encoder runs on full buffer each iteration
  - Left context encoder frames are STRIPPED — only chunk frames are decoded
  - Decoder LSTM state carries forward between chunks
  - Each token is decoded exactly once (TDT durations handle frame advancement)

Model: parakeet-tdt-0.6b-v3-onnx-fp32/fp32
  - encoder.onnx: audio_signal [B, 128, T], length [B] → outputs [B, 1024, T'], encoded_lengths [B]
  - decoder.onnx: targets [B, T] i64, target_length_orig [B] i64, h_in [2, B, 640], c_in [2, B, 640]
                   → decoder_output [B, 640, T'], target_length [B], h_out [2, B, 640], c_out [2, B, 640]
  - joint.onnx:   encoder_output [B, T, 1024], decoder_output [B, T', 640]
                   → joint_output [B, T, T', 8198]  (8193 vocab + 5 TDT durations)

Usage:
    python parakeet_onnx_streaming.py
"""

import numpy as np
import onnxruntime as ort
import os
import sys
import time

# ═══════════════════════════════════════════════════════════════════════════════
#  Model constants (from audio_processor_config.json, tdt_config.json, genai_config.json)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000
VOCAB_SIZE = 8193
BLANK_ID = 8192
NUM_DURATIONS = 5
DURATIONS = [0, 1, 2, 3, 4]
MAX_SYMBOLS_PER_STEP = 10

# Mel / Preprocessor (audio_processor_config.json)
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 2**-24  # NeMo FilterbankFeatures default (not the ONNX config 1e-10)
FMIN = 0
FMAX = 8000

# Encoder
ENCODER_SUBSAMPLING = 8  # from genai_config.json speech.subsampling_factor

# ═══════════════════════════════════════════════════════════════════════════════
#  Streaming context — NeMo recommended defaults for long file transcription
# ═══════════════════════════════════════════════════════════════════════════════

LEFT_CONTEXT_SEC = 9.0
CHUNK_SEC = 0.8
RIGHT_CONTEXT_SEC = 1.6

# Align to encoder frame boundaries (same as NeMo's make_divisible_by)
ENCODER_FRAME_SAMPLES = HOP_LENGTH * ENCODER_SUBSAMPLING  # 1280
LEFT_SAMPLES = int(LEFT_CONTEXT_SEC * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES  # 160000
CHUNK_SAMPLES = int(CHUNK_SEC * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES        # 32000
RIGHT_SAMPLES = int(RIGHT_CONTEXT_SEC * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES  # 32000

# ═══════════════════════════════════════════════════════════════════════════════
#  Mel filterbank and Hann window (computed once, cached)
# ═══════════════════════════════════════════════════════════════════════════════

_mel_basis_cache = None


def _get_mel_basis():
    """Librosa mel filterbank [128, 257], computed once."""
    global _mel_basis_cache
    if _mel_basis_cache is None:
        import librosa
        _mel_basis_cache = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX
        ).astype(np.float32)
    return _mel_basis_cache


_hann_window_cache = None


def _get_hann_window():
    """Symmetric Hann window of length WIN_LENGTH, computed once."""
    global _hann_window_cache
    if _hann_window_cache is None:
        import torch
        _hann_window_cache = torch.hann_window(WIN_LENGTH, periodic=False)
    return _hann_window_cache


# ═══════════════════════════════════════════════════════════════════════════════
#  Mel feature computation — matches NeMo's AudioToMelSpectrogramPreprocessor
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mel_features(audio_np):
    """
    Compute log-mel features matching NeMo's FilterbankFeatures.forward() exactly.

    Pipeline (NeMo):
      1. Preemphasis: x[0] unchanged, x[n] = x[n] - 0.97*x[n-1] for n>0
      2. STFT (n_fft=512, hop=160, win=400, center=True, symmetric Hann, zero-pad)
      3. Magnitude: sqrt(re^2 + im^2)
      4. Power: magnitude^2.0
      5. Mel filterbank (128 bins, fmin=0, fmax=8000)
      6. Log (with 2^-24 guard — NeMo default)
      7. Truncate to seq_len // hop_length valid frames

    Args:
        audio_np: float32 audio samples

    Returns:
        [T, 128] numpy float32 (unnormalized — before per_feature normalization)
    """
    import torch

    seq_len = len(audio_np)
    x = torch.from_numpy(audio_np).unsqueeze(0).float()  # [1, N]

    # 1. Preemphasis (NeMo style: first sample unchanged, no cross-window continuity)
    x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - PREEMPH * x[:, :-1]), dim=1)

    # 2. STFT (center=True, pad_mode="constant" matches NeMo default)
    window = _get_hann_window()
    stft_out = torch.stft(
        x.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        center=True, window=window, return_complex=True,
        pad_mode="constant", normalized=False
    )  # [257, T_stft] complex

    # 3-4. Magnitude then power (NeMo: sqrt(re^2+im^2) then .pow(mag_power=2.0))
    x = torch.view_as_real(stft_out)  # [257, T_stft, 2]
    x = torch.sqrt(x.pow(2).sum(-1))  # magnitude [257, T_stft]
    x = x.pow(2.0)  # power

    # 5. Mel filterbank
    mel_basis = torch.from_numpy(_get_mel_basis())  # [128, 257]
    mel_spec = torch.matmul(mel_basis, x)  # [128, T_stft]

    # 6. Log with guard (NeMo default: log_zero_guard_type="add", value=2^-24)
    mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD)

    # 7. Truncate to valid frames (NeMo: get_seq_len = seq_len // hop_length)
    valid_frames = seq_len // HOP_LENGTH
    mel_spec = mel_spec[:, :valid_frames]  # [128, valid_frames]

    return mel_spec.T.numpy()  # [T, 128]


def normalize_mel_per_feature(raw_mel):
    """
    Per-feature normalization with Bessel's correction.
    Matches NeMo's normalize_batch with per_feature mode.

    Input:  [T, 128]  (time, features)
    Output: [128, T]  (features, time) — ready for encoder
    """
    n = raw_mel.shape[0]  # number of time frames
    mean = raw_mel.mean(axis=0)  # [128]
    if n > 1:
        var = np.sum((raw_mel - mean) ** 2, axis=0) / (n - 1)  # Bessel's correction
    else:
        var = np.zeros_like(mean)
    std = np.sqrt(var) + 1e-5
    normalized = (raw_mel - mean) / std
    return normalized.T.astype(np.float32)  # [128, T]


def normalize_mel_with_stats(raw_mel, mean, std):
    """
    Normalize mel using externally provided mean/std statistics.

    Input:  raw_mel [T, 128], mean [128], std [128]
    Output: [128, T]  (features, time) — ready for encoder
    """
    normalized = (raw_mel - mean) / std
    return normalized.T.astype(np.float32)  # [128, T]


class RunningMelStats:
    """
    Running (online) accumulator for per-feature mel normalization statistics.
    Uses Welford's algorithm for numerically stable online variance.
    Provides causal normalization: only uses audio received so far.
    """

    def __init__(self, n_features=N_MELS):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)  # sum of squared diffs

    def update(self, raw_mel):
        """
        Add new mel frames to running statistics.

        Args:
            raw_mel: [T, 128] new mel frames (NOT previously seen frames)
        """
        for i in range(raw_mel.shape[0]):
            self.n += 1
            delta = raw_mel[i].astype(np.float64) - self.mean
            self.mean += delta / self.n
            delta2 = raw_mel[i].astype(np.float64) - self.mean
            self.m2 += delta * delta2

    def get_mean_std(self):
        """
        Get current mean and std (with Bessel's correction).

        Returns:
            (mean [128], std [128]) as float32
        """
        mean = self.mean.astype(np.float32)
        if self.n > 1:
            var = (self.m2 / (self.n - 1)).astype(np.float32)
        else:
            var = np.zeros_like(mean)
        std = np.sqrt(var) + 1e-5
        return mean, std


# ═══════════════════════════════════════════════════════════════════════════════
#  Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

def load_vocab(model_dir):
    """Load vocabulary from vocab.txt (one token per line, 0-indexed)."""
    vocab = {}
    vocab_path = os.path.join(model_dir, "vocab.txt")
    with open(vocab_path) as f:
        for idx, line in enumerate(f):
            vocab[idx] = line.strip()
    return vocab


def tokens_to_text(tokens, vocab):
    """Convert token IDs to text using SentencePiece ▁ → space replacement."""
    pieces = []
    for t in tokens:
        piece = vocab.get(t, "")
        pieces.append(piece.replace("\u2581", " "))
    return "".join(pieces)


# ═══════════════════════════════════════════════════════════════════════════════
#  Parakeet ONNX inference class
# ═══════════════════════════════════════════════════════════════════════════════

class ParakeetONNX:
    """
    Parakeet TDT 0.6B v3 ONNX FP32 inference — streaming and batch.

    Implements the exact same algorithm as NVIDIA NeMo's
    speech_to_text_streaming_infer_rnnt.py.
    """

    def __init__(self, model_dir, use_cuda=True):
        self.model_dir = model_dir
        self.vocab = load_vocab(model_dir)

        providers = []
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "encoder.onnx"), sess_options=so, providers=providers
        )
        self.decoder = ort.InferenceSession(
            os.path.join(model_dir, "decoder.onnx"), sess_options=so, providers=providers
        )
        self.joiner = ort.InferenceSession(
            os.path.join(model_dir, "joint.onnx"), sess_options=so, providers=providers
        )

        print(f"Loaded ONNX models from {model_dir}")
        print(f"  Provider: {self.encoder.get_providers()}")
        print(f"  Vocab size: {len(self.vocab)}")

    # ── Decoder state initialization ──

    def _init_decoder_state(self):
        """Initialize decoder LSTM with blank token input."""
        h = np.zeros((2, 1, 640), dtype=np.float32)
        c = np.zeros((2, 1, 640), dtype=np.float32)
        d = self.decoder.run(None, {
            "targets": np.array([[BLANK_ID]], dtype=np.int64),
            "target_length_orig": np.array([1], dtype=np.int64),
            "h_in": h,
            "c_in": c,
        })
        dec_out = d[0][:, :, -1:]  # [1, 640, 1]
        h_out = d[2]               # [2, 1, 640]
        c_out = d[3]               # [2, 1, 640]
        return dec_out, h_out, c_out

    # ── TDT greedy decoding ──

    def _tdt_greedy_decode(self, encoded, decode_start, decode_end, dec_out, h, c):
        """
        TDT greedy decode on encoder frames [decode_start, decode_end).
        Decoder state (dec_out, h, c) carries forward between chunks.

        Args:
            encoded:      encoder output [1, 1024, T] (raw from encoder, NOT transposed)
            decode_start: first encoder frame index to decode
            decode_end:   one past last encoder frame index to decode
            dec_out:      decoder output [1, 640, 1]
            h, c:         LSTM states [2, 1, 640]

        Returns:
            (tokens, dec_out, h, c)
        """
        tokens = []
        symbols_this_frame = 0
        t = decode_start

        while t < decode_end:
            # Encoder frame: [1, 1024, 1] → transpose to [1, 1, 1024] for joiner
            enc_frame = encoded[:, :, t:t + 1].transpose(0, 2, 1)  # [1, 1, 1024]
            # Decoder output: [1, 640, 1] → transpose to [1, 1, 640] for joiner
            dec_frame = dec_out.transpose(0, 2, 1)  # [1, 1, 640]

            # Run joiner → [1, 1, 1, 8198]
            j = self.joiner.run(None, {
                "encoder_output": enc_frame,
                "decoder_output": dec_frame,
            })
            logits = j[0].squeeze()  # [8198]

            # Token prediction (first 8193 logits = vocab)
            tok = int(np.argmax(logits[:VOCAB_SIZE]))
            # Duration prediction (next 5 logits = TDT durations [0,1,2,3,4])
            skip = int(np.argmax(logits[VOCAB_SIZE:VOCAB_SIZE + NUM_DURATIONS]))

            if tok != BLANK_ID:
                tokens.append(tok)
                symbols_this_frame += 1
                # Update decoder state with emitted token
                d = self.decoder.run(None, {
                    "targets": np.array([[tok]], dtype=np.int64),
                    "target_length_orig": np.array([1], dtype=np.int64),
                    "h_in": h,
                    "c_in": c,
                })
                dec_out = d[0][:, :, -1:]  # [1, 640, 1]
                h = d[2]                    # [2, 1, 640]
                c = d[3]                    # [2, 1, 640]

            # Handle duration: skip > 0 means advance by 'skip' frames
            if skip > 0:
                symbols_this_frame = 0

            # Safety: force advance if too many symbols at one frame
            if symbols_this_frame >= MAX_SYMBOLS_PER_STEP:
                symbols_this_frame = 0
                skip = 1

            # Force advance if blank with duration 0 (prevent infinite loop)
            if tok == BLANK_ID and skip == 0:
                symbols_this_frame = 0
                skip = 1

            t += skip

        return tokens, dec_out, h, c

    # ── Streaming (chunked) inference ──

    def transcribe_streaming(self, audio,
                              chunk_sec=CHUNK_SEC,
                              left_sec=LEFT_CONTEXT_SEC,
                              right_sec=RIGHT_CONTEXT_SEC,
                              live=False):
        """
        Streaming chunked inference matching NeMo's algorithm.

        When live=True, prints each chunk's text as it's decoded (real-time streaming).

        Args:
            audio:     float32 numpy array, mono, 16 kHz
            chunk_sec: chunk size in seconds
            left_sec:  left context in seconds
            right_sec: right context in seconds
            live:      if True, print text chunk by chunk as decoded

        Returns:
            Transcription string
        """
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Align context sizes to encoder frame boundaries
        left_samples = int(left_sec * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES
        chunk_samples = int(chunk_sec * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES
        right_samples = int(right_sec * SAMPLE_RATE // ENCODER_FRAME_SAMPLES) * ENCODER_FRAME_SAMPLES

        if not live:
            print(f"  Context (sec): left={left_samples / SAMPLE_RATE:.2f}, "
                  f"chunk={chunk_samples / SAMPLE_RATE:.2f}, "
                  f"right={right_samples / SAMPLE_RATE:.2f}")
            print(f"  Theoretical latency: {(chunk_samples + right_samples) / SAMPLE_RATE:.2f}s")

        # Initialize decoder state
        dec_out, h, c = self._init_decoder_state()

        all_tokens = []
        chunk_start = 0
        chunk_idx = 0

        while chunk_start < len(audio):
            chunk_end = min(chunk_start + chunk_samples, len(audio))
            is_last = chunk_end >= len(audio)

            # ── Build encoder window: [left_context | chunk | right_context] ──
            win_left = max(0, chunk_start - left_samples)
            win_right = min(chunk_end + right_samples, len(audio))
            if is_last:
                target_buf_size = left_samples + chunk_samples + right_samples
                actual_size = win_right - win_left
                if actual_size < target_buf_size:
                    win_left = max(0, win_right - target_buf_size)
            window_audio = audio[win_left:win_right]

            # ── Compute raw mel features on the window ──
            raw_mel = compute_mel_features(window_audio)  # [T, 128]

            # ── Per-window normalization (matches NeMo exactly) ──
            mel_norm = normalize_mel_per_feature(raw_mel)  # [128, T]

            # ── Run encoder ──
            mel_in = mel_norm[np.newaxis, :, :]  # [1, 128, T]
            length = np.array([mel_norm.shape[1]], dtype=np.int64)
            enc_out = self.encoder.run(None, {"audio_signal": mel_in, "length": length})
            encoded = enc_out[0]     # [1, 1024, T']
            enc_total = int(enc_out[1][0])

            # ── Strip left context encoder frames ──
            left_ctx_samples = chunk_start - win_left
            left_ctx_mel = left_ctx_samples // HOP_LENGTH
            left_enc = left_ctx_mel // ENCODER_SUBSAMPLING

            # ── Determine decode range ──
            if is_last:
                decode_start = left_enc
                decode_end = enc_total
            else:
                decode_start = left_enc
                chunk_actual = chunk_end - chunk_start
                chunk_mel = chunk_actual // HOP_LENGTH
                chunk_enc = chunk_mel // ENCODER_SUBSAMPLING
                decode_end = min(left_enc + chunk_enc, enc_total)

            # ── DEBUG: dump mel + encoder output for first 10 chunks ──
            if chunk_idx < 10:
                dump_dir = os.path.expanduser("~/debug_py")
                os.makedirs(dump_dir, exist_ok=True)
                np.save(f"{dump_dir}/chunk{chunk_idx}_raw_mel.npy", raw_mel)      # [T, 128]
                np.save(f"{dump_dir}/chunk{chunk_idx}_mel_norm.npy", mel_norm)    # [128, T]
                np.save(f"{dump_dir}/chunk{chunk_idx}_enc_out.npy", encoded)      # [1, 1024, T']
                print(f"  [PY DEBUG chunk {chunk_idx}] chunk_start={chunk_start} chunk_end={chunk_end} "
                      f"win_left={win_left} win_right={win_right} "
                      f"window_samples={len(window_audio)} mel_frames={mel_norm.shape[1]} "
                      f"enc_frames={enc_total} left_enc={left_enc} "
                      f"decode=[{decode_start},{decode_end})")

            if decode_end <= decode_start:
                chunk_start = chunk_end
                chunk_idx += 1
                continue

            # ── TDT greedy decode ──
            chunk_tokens, dec_out, h, c = self._tdt_greedy_decode(
                encoded, decode_start, decode_end, dec_out, h, c
            )
            all_tokens.extend(chunk_tokens)

            # ── Live output: print each chunk's text as decoded ──
            if live:
                chunk_text = tokens_to_text(chunk_tokens, self.vocab)
                timestamp = chunk_start / SAMPLE_RATE
                if chunk_text.strip():
                    sys.stdout.write(f"[{timestamp:6.1f}s] {chunk_text}\n")
                    sys.stdout.flush()

            # ── Advance to next chunk ──
            chunk_start = chunk_end
            chunk_idx += 1

        return tokens_to_text(all_tokens, self.vocab)

    # ── Batch (offline) inference ──

    def transcribe_batch(self, audio):
        """
        Full-sequence (offline/batch) inference.

        Same mel + encoder + TDT decode, but over the entire utterance at once.

        Args:
            audio: float32 numpy array, mono, 16 kHz

        Returns:
            Transcription string
        """
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Compute mel + normalize over full utterance
        raw_mel = compute_mel_features(audio)  # [T, 128]
        mel_norm = normalize_mel_per_feature(raw_mel)  # [128, T]

        # Run encoder on full utterance
        mel_in = mel_norm[np.newaxis, :, :]  # [1, 128, T]
        length = np.array([mel_norm.shape[1]], dtype=np.int64)
        enc_out = self.encoder.run(None, {"audio_signal": mel_in, "length": length})
        encoded = enc_out[0]     # [1, 1024, T']
        enc_len = int(enc_out[1][0])

        # Initialize decoder
        dec_out, h, c = self._init_decoder_state()

        # Decode full sequence
        tokens, _, _, _ = self._tdt_greedy_decode(
            encoded, 0, enc_len, dec_out, h, c
        )
        return tokens_to_text(tokens, self.vocab)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — run both streaming and batch on mars_16k.wav
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    model_dir = os.path.expanduser("~/parakeet-tdt-0.6b-v3-onnx-fp32/fp32")
    audio_path = os.path.expanduser("~/speech.wav")

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # ── Load audio ──
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype='float32')
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / SAMPLE_RATE
    print(f"Audio: {audio_path}")
    print(f"  Duration: {duration:.2f}s, Samples: {len(audio)}, SR: {SAMPLE_RATE}")
    print()

    # ── Load model ──
    asr = ParakeetONNX(model_dir, use_cuda=True)
    print()

    # ── Batch (offline) inference ──
    print("=" * 70)
    print("BATCH (OFFLINE) INFERENCE")
    print("=" * 70)
    t0 = time.time()
    #text_batch = asr.transcribe_batch(audio)
    t1 = time.time()
    print(f"  Time: {t1 - t0:.3f}s")
    print(f"  RTFx: {duration / (t1 - t0):.1f}x")
    #print(f"  Text: {text_batch}")
    print()

    # ── Streaming inference ──
    print("=" * 70)
    print("STREAMING INFERENCE (live, chunk by chunk)")
    print("=" * 70)
    t0 = time.time()
    text_stream = asr.transcribe_streaming(audio, live=True)
    t1 = time.time()
    print(f"\n  Time: {t1 - t0:.3f}s")
    print(f"  RTFx: {duration / (t1 - t0):.1f}x")
    print(f"  Full text: {text_stream}")
    print()

    # ── Comparison ──
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()