"""
Generate ASR-transcribed text via a TTS → noise injection → ASR pipeline.

Pipeline per row
----------------
1. TTS : edge-tts synthesises speech from the text
2. Mix : torchaudio mixes speech + noise file at a target SNR (dB)
3. ASR : faster-whisper transcribes the noisy audio back to text

Output CSVs have the same format as text_noise files (text column replaced
by ASR transcription) and are compatible with get_llm_responses.py.

Usage
-----
  python add_audio_noise.py                         # all noise types × all SNR levels
  python add_audio_noise.py --noise-type white cafe # two noise types
  python add_audio_noise.py --snr 0 5 10 20         # subset of SNR levels (dB)
  python add_audio_noise.py --noise-type white --snr 0 5 10

Noise files  (place in Dataset/noise/ — run download_noise.py first)
-----------
  white_noise.wav, pink_noise.wav, cafe_noise.wav,
  street_noise.wav, crowd_noise.wav, traffic_noise.wav

Output
------
  Dataset/audio_noise/{noise_type}/hate_speech_{N}_clean.csv       (no noise)
  Dataset/audio_noise/{noise_type}/hate_speech_{N}_snr_p30db.csv   (SNR = +30 dB)
  ...
  Dataset/audio_noise/{noise_type}/hate_speech_{N}_snr_m20db.csv   (SNR = −20 dB)
"""

import argparse
import asyncio
import glob
import math
import os
import shutil
import tempfile

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
DATASET_DIR  = "./Dataset"
NOISE_DIR    = "./Dataset/noise"
OUTPUT_DIR   = "./Dataset/audio_noise"

# 11 environments: None = clean (TTS → ASR, no noise added)
SNR_LEVELS   = [None, 30, 20, 15, 10, 5, 0, -5, -10, -15, -20]
NOISE_TYPES  = ["white", "pink", "cafe", "street", "crowd", "traffic"]

TARGET_SR    = 16_000          # resample all audio to 16 kHz
TTS_VOICE    = "en-US-GuyNeural"
WHISPER_SIZE = "base"          # tiny / base / small / medium / large
TTS_WORKERS  = 8               # parallel TTS coroutines
# ==============================================================================


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _snr_label(snr):
    if snr is None:
        return "clean"
    sign = "m" if snr < 0 else "p"
    return f"snr_{sign}{abs(snr):02d}db"


def _find_input_path():
    pattern = os.path.join(DATASET_DIR, "hate_speech_binary_*.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No sample CSV found in {DATASET_DIR}. Run download_data.py first."
        )
    if len(candidates) == 1:
        return candidates[0]
    print("Multiple sample files found:")
    for i, p in enumerate(candidates):
        print(f"  [{i}] {os.path.basename(p)}")
    idx = int(input("Enter the number of the file to use: ").strip())
    return candidates[idx]


def _load_mono_16k(path: str) -> torch.Tensor:
    """Load audio → mono float32 → resample to TARGET_SR."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav                    # (1, T)


def _loop_to(noise: torch.Tensor, target_len: int) -> torch.Tensor:
    """Tile noise to match target_len samples."""
    if noise.shape[-1] >= target_len:
        return noise[..., :target_len]
    reps = math.ceil(target_len / noise.shape[-1])
    return noise.repeat(1, reps)[..., :target_len]


def mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Return speech + noise at the requested SNR (dB)."""
    noise   = _loop_to(noise, speech.shape[-1])
    s_rms   = speech.norm(p=2) / math.sqrt(speech.numel()) + 1e-10
    n_rms   = noise.norm(p=2)  / math.sqrt(noise.numel())  + 1e-10
    scale   = s_rms / (10 ** (snr_db / 20)) / n_rms
    return speech + noise * scale


# ------------------------------------------------------------------------------
# TTS  (edge-tts, async)
# ------------------------------------------------------------------------------

async def _tts_one(text: str, out_mp3: str, voice: str):
    import edge_tts
    comm = edge_tts.Communicate(str(text), voice=voice)
    await comm.save(out_mp3)


async def _tts_all(texts, mp3_paths, voice=TTS_VOICE):
    sem = asyncio.Semaphore(TTS_WORKERS)

    async def _bounded(t, p):
        async with sem:
            await _tts_one(t, p, voice)

    await asyncio.gather(*[_bounded(t, p) for t, p in zip(texts, mp3_paths)])


# ------------------------------------------------------------------------------
# ASR  (faster-whisper)
# ------------------------------------------------------------------------------

def load_whisper(size=WHISPER_SIZE):
    from faster_whisper import WhisperModel
    print(f"  Loading Whisper '{size}' model ...")
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(model, wav_path: str) -> str:
    segments, _ = model.transcribe(wav_path, beam_size=5, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def run_pipeline(noise_types=None, snr_levels=None):
    noise_types = noise_types or NOISE_TYPES
    snr_levels  = snr_levels  or SNR_LEVELS

    # --- Load input dataset ---
    input_path = _find_input_path()
    datasize   = os.path.splitext(os.path.basename(input_path))[0].split("_")[-1]
    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="latin1")
    texts = df["text"].fillna("").tolist()
    n     = len(texts)
    print(f"Loaded {n} rows from {input_path}")

    # --- Step 1: TTS — one MP3 per row ---
    tts_dir   = tempfile.mkdtemp(prefix="tts_")
    mp3_paths = [os.path.join(tts_dir, f"row_{i:04d}.mp3") for i in range(n)]
    print(f"\n[1/3] TTS synthesis ({TTS_VOICE}) ...")
    asyncio.run(_tts_all(texts, mp3_paths))

    # Convert MP3 → 16 kHz mono WAV (needed for torchaudio noise mixing)
    speech_wavs = []
    wav_paths   = []
    for i, mp3 in enumerate(tqdm(mp3_paths, desc="  Loading TTS audio")):
        wav  = _load_mono_16k(mp3)
        path = mp3.replace(".mp3", ".wav")
        torchaudio.save(path, wav, TARGET_SR)
        speech_wavs.append(wav)
        wav_paths.append(path)

    # --- Step 2: Load Whisper ---
    print("\n[2/3] Loading ASR model ...")
    whisper = load_whisper()

    # --- Step 3: For each noise type × SNR level, inject noise + transcribe ---
    print("\n[3/3] Noise injection + ASR transcription ...")

    # Cache clean transcriptions (same across all noise types)
    clean_transcriptions = None

    for noise_type in noise_types:
        noise_file = os.path.join(NOISE_DIR, f"{noise_type}_noise.wav")
        if not os.path.exists(noise_file):
            print(f"\n  Skip '{noise_type}': {noise_file} not found. Run download_noise.py.")
            continue

        noise_wav = _load_mono_16k(noise_file)
        out_dir   = os.path.join(OUTPUT_DIR, noise_type)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  Noise type: {noise_type}")
        for snr in tqdm(snr_levels, desc=f"    SNR levels"):
            label    = _snr_label(snr)
            out_path = os.path.join(out_dir, f"hate_speech_{datasize}_{label}.csv")

            if os.path.exists(out_path):
                tqdm.write(f"      Skip (exists): {os.path.basename(out_path)}")
                continue

            if snr is None:
                # Clean: TTS → ASR (no noise) — compute once, reuse across noise types
                if clean_transcriptions is None:
                    clean_transcriptions = []
                    for wav_path in tqdm(wav_paths, desc="      Clean ASR", leave=False):
                        clean_transcriptions.append(transcribe(whisper, wav_path))
                transcriptions = clean_transcriptions
            else:
                transcriptions = []
                for speech, wav_path in tqdm(
                    zip(speech_wavs, wav_paths), total=n,
                    desc=f"      SNR {snr:+d} dB", leave=False
                ):
                    noisy     = mix_at_snr(speech, noise_wav, snr)
                    noisy_wav = wav_path.replace(".wav", f"_{label}.wav")
                    torchaudio.save(noisy_wav, noisy, TARGET_SR)
                    transcriptions.append(transcribe(whisper, noisy_wav))
                    os.remove(noisy_wav)

            out_df          = df.copy()
            out_df["text"]  = transcriptions
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            tqdm.write(f"      Saved: {os.path.basename(out_path)}")

    # Cleanup TTS temp files
    shutil.rmtree(tts_dir, ignore_errors=True)
    print(f"\nDone. Results in {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="TTS → noise injection → ASR pipeline"
    )
    parser.add_argument(
        "--noise-type", nargs="+", choices=NOISE_TYPES, default=None,
        metavar="TYPE",
        help=f"Noise type(s) to process (default: all). Choices: {NOISE_TYPES}",
    )
    parser.add_argument(
        "--snr", nargs="+", type=float, default=None,
        metavar="DB",
        help="SNR levels in dB (default: all 11 levels including clean). "
             "Example: --snr 0 5 10 20",
    )
    args = parser.parse_args()

    snr_levels = None
    if args.snr is not None:
        # Always include clean (None) as the zero-noise baseline
        snr_levels = [None] + [int(s) if s == int(s) else s for s in args.snr]

    run_pipeline(noise_types=args.noise_type, snr_levels=snr_levels)


if __name__ == "__main__":
    main()
