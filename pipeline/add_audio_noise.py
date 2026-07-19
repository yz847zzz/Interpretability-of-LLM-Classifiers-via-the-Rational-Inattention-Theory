"""
Three-stage audio noise pipeline

Stage 1  --step tts
    Text → TTS speech (edge-tts)
    Output: Dataset/audio_tts/{id}.wav          (200 files)

Stage 2  --step mix
    Clean speech + noise file @ SNR → noisy audio
    Each SNR level gets its own folder containing, per row:
        {id}_clean.wav   — original TTS speech (copy for easy comparison)
        {id}_noisy.wav   — clean speech + noise at this SNR
    Output: Dataset/audio_noisy/{noise_type}/snr_{label}/{id}_clean.wav
            Dataset/audio_noisy/{noise_type}/snr_{label}/{id}_noisy.wav

Stage 3  --step asr
    Noisy audio → ASR transcription → CSV
    Output: Dataset/audio_noise/{noise_type}/hate_speech_{N}_{label}.csv
            same columns as text_noise CSVs + 'id' column

Usage
-----
    python add_audio_noise.py                              # all stages
    python add_audio_noise.py --step tts
    python add_audio_noise.py --step mix --noise-type white cafe --snr 0 5 10
    python add_audio_noise.py --step asr --noise-type white --snr 0 5 10

Noise files (Dataset/noise/) — run download_noise.py first:
    white_noise.wav  pink_noise.wav  cafe_noise.wav
    street_noise.wav  crowd_noise.wav  traffic_noise.wav
"""


import argparse
import asyncio
import glob
import math
import os
import av
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io.wavfile import write as _wav_write
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
DATASET_DIR  = "./Dataset"
NOISE_DIR    = "./Dataset/noise"
TTS_DIR      = "./Dataset/audio_original"   # Stage 1: clean speech WAVs
MIX_DIR      = "./Dataset/audio_noisy"     # Stage 2: clean + noisy WAVs per SNR folder
OUTPUT_DIR   = "./Dataset/text_with_audio_noisy"  # Stage 3: ASR transcription CSVs

SNR_LEVELS   = [40, 30, 20, 15, 10, 5, 4, 3, 2, 1, 0, -1, -2, -3, -5, -6, -7, -8, -10, -15, -20]   # dB
NOISE_TYPES  = ["white", "babble", "cafe", "crowd"]

TARGET_SR    = 16_000
TTS_VOICE    = "en-US-GuyNeural"
WHISPER_SIZE = "base"         # tiny / base / small / medium / large
TTS_WORKERS  = 8
# ==============================================================================


# ------------------------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------------------------

def _snr_label(snr: int) -> str:
    sign = "m" if snr < 0 else "p"
    return f"snr_{sign}{abs(snr):02d}db"


def _find_input_path() -> str:
    candidates = sorted(glob.glob(os.path.join(DATASET_DIR, "hate_speech_binary_*.csv")))
    if not candidates:
        raise FileNotFoundError(
            f"No sample CSV in {DATASET_DIR}. Run download_data.py first."
        )
    if len(candidates) == 1:
        return candidates[0]
    print("Multiple sample files found:")
    for i, p in enumerate(candidates):
        print(f"  [{i}] {os.path.basename(p)}")
    return candidates[int(input("Choose file number: ").strip())]


def _load_df() -> pd.DataFrame:
    path = _find_input_path()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    if "id" not in df.columns:
        df.insert(0, "id", [f"{i:04d}" for i in range(len(df))])
    return df, path


def _load_mono_16k(path: str) -> torch.Tensor:
    from scipy.io.wavfile import read as wav_read
    sr, data = wav_read(path)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data).unsqueeze(0)  # (1, T)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav   # (1, T)


def _sample_noise(noise: torch.Tensor, length: int, rng: np.random.Generator) -> torch.Tensor:
    """Randomly crop `length` samples from noise; tile if noise is shorter."""
    n = noise.shape[-1]
    if n < length:
        noise = noise.repeat(1, math.ceil(length / n))
        n = noise.shape[-1]
    max_offset = n - length
    offset = int(rng.integers(0, max_offset + 1))
    return noise[..., offset:offset + length]


def _mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float,
                rng: np.random.Generator) -> torch.Tensor:
    segment  = _sample_noise(noise, speech.shape[-1], rng)
    # Peak-normalise speech before computing RMS to avoid silence deflating the reference
    peak     = speech.abs().max() + 1e-10
    s_rms    = (speech / peak).norm(p=2) / math.sqrt(speech.numel()) + 1e-10
    n_rms    = segment.norm(p=2) / math.sqrt(segment.numel()) + 1e-10
    gain     = s_rms / (10 ** (snr_db / 20)) / n_rms
    return speech + segment * gain


# ------------------------------------------------------------------------------
# Stage 1 — Text → TTS WAV
# ------------------------------------------------------------------------------

def _mp3_to_wav(mp3_path: str, wav_path: str):
    """Decode MP3 via PyAV, resample to TARGET_SR, save as mono WAV."""
    frames = []
    with av.open(mp3_path) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SR)
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray()[0])
    audio = np.concatenate(frames).astype(np.float32)
    _wav_write(wav_path, TARGET_SR, audio)


async def _tts_one(text: str, mp3_path: str, voice: str):
    import edge_tts
    await edge_tts.Communicate(str(text), voice=voice).save(mp3_path)


async def _tts_all_async(rows: list[tuple], voice: str):
    """rows: list of (id, text, wav_path)"""
    sem = asyncio.Semaphore(TTS_WORKERS)

    async def _one(row_id: str, text: str, wav_path: str):
        if os.path.exists(wav_path):
            return
        mp3_path = wav_path.replace(".wav", ".mp3")
        async with sem:
            await _tts_one(text, mp3_path, voice)
        _mp3_to_wav(mp3_path, wav_path)
        os.remove(mp3_path)

    await asyncio.gather(*[_one(r[0], r[1], r[2]) for r in rows])


def stage_tts():
    df, path = _load_df()
    n = len(df)
    print(f"Stage 1: TTS synthesis — {n} rows → {TTS_DIR}/")
    os.makedirs(TTS_DIR, exist_ok=True)

    rows = [
        (row["id"], row["text"], os.path.join(TTS_DIR, f"{row['id']}.wav"))
        for _, row in df.iterrows()
    ]
    existing = sum(1 for _, _, p in rows if os.path.exists(p))
    if existing == n:
        print(f"  All {n} WAV files already exist — skipping.")
        return

    asyncio.run(_tts_all_async(rows, TTS_VOICE))
    done = sum(1 for _, _, p in rows if os.path.exists(p))
    print(f"  Done. {done}/{n} WAV files in {TTS_DIR}/")


# ------------------------------------------------------------------------------
# Stage 2 — Clean + noise @ SNR → noisy WAVs organised by SNR folder
# ------------------------------------------------------------------------------

def stage_mix(noise_types=None, snr_levels=None):
    noise_types = noise_types or NOISE_TYPES
    snr_levels  = snr_levels  or SNR_LEVELS

    tts_wavs = sorted(glob.glob(os.path.join(TTS_DIR, "*.wav")))
    if not tts_wavs:
        raise FileNotFoundError(f"No WAVs in {TTS_DIR}. Run --step tts first.")
    n = len(tts_wavs)

    print(f"Stage 2: Mixing {n} speeches × {len(noise_types)} noise types "
          f"× {len(snr_levels)} SNR levels → {MIX_DIR}/")

    for noise_type in noise_types:
        noise_file = os.path.join(NOISE_DIR, f"{noise_type}_noise.wav")
        if not os.path.exists(noise_file):
            print(f"  Skip '{noise_type}': {noise_file} not found. Run download_noise.py.")
            continue

        noise_wav = _load_mono_16k(noise_file)

        for snr in tqdm(snr_levels, desc=f"  {noise_type}"):
            label   = _snr_label(snr)
            snr_dir = os.path.join(MIX_DIR, noise_type, label)
            os.makedirs(snr_dir, exist_ok=True)

            for idx, tts_path in enumerate(tts_wavs):
                row_id    = os.path.splitext(os.path.basename(tts_path))[0]
                noisy_out = os.path.join(snr_dir, f"{row_id}.wav")

                if os.path.exists(noisy_out):
                    continue

                speech = _load_mono_16k(tts_path)
                rng    = np.random.default_rng(seed=idx)
                noisy  = _mix_at_snr(speech, noise_wav, snr, rng)
                _wav_write(noisy_out, TARGET_SR, noisy.squeeze(0).numpy())

    print("  Done.")


# ------------------------------------------------------------------------------
# Stage 3 — Noisy WAVs → ASR → CSV
# ------------------------------------------------------------------------------

def _load_whisper():
    from faster_whisper import WhisperModel
    print(f"  Loading Whisper '{WHISPER_SIZE}' ...")
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


def _transcribe(model, wav_path: str) -> str:
    segs, _ = model.transcribe(wav_path, beam_size=5, language="en")
    return " ".join(s.text.strip() for s in segs).strip()


def stage_asr(noise_types=None, snr_levels=None):
    noise_types = noise_types or NOISE_TYPES
    snr_levels  = snr_levels  or SNR_LEVELS

    df, path = _load_df()
    n        = len(df)
    datasize = os.path.splitext(os.path.basename(path))[0].split("_")[-1]

    print(f"Stage 3: ASR transcription → {OUTPUT_DIR}/")
    whisper = _load_whisper()

    for noise_type in noise_types:
        out_dir = os.path.join(OUTPUT_DIR, noise_type)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  Noise type: {noise_type}")
        for snr in tqdm(snr_levels, desc=f"    SNR levels"):
            label    = _snr_label(snr)
            snr_dir  = os.path.join(MIX_DIR, noise_type, label)
            out_csv  = os.path.join(out_dir, f"hate_speech_{datasize}_{label}.csv")

            if os.path.exists(out_csv):
                tqdm.write(f"      Skip (exists): {os.path.basename(out_csv)}")
                continue

            if not os.path.isdir(snr_dir):
                tqdm.write(f"      Skip {label}: {snr_dir} not found. Run --step mix.")
                continue

            # build id→wav_path map; filenames are {id}.wav
            wav_map = {
                os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(snr_dir, "*.wav"))
            }
            missing = [row["id"] for _, row in df.iterrows() if str(row["id"]) not in wav_map]
            if missing:
                tqdm.write(
                    f"      Skip {label}: {len(missing)} WAVs missing "
                    f"(e.g. {missing[0]}). Run --step mix."
                )
                continue

            # transcribe in df row order — strict id correspondence
            transcriptions = [
                _transcribe(whisper, wav_map[str(row["id"])])
                for _, row in tqdm(df.iterrows(), total=n,
                                   desc=f"      {label}", leave=False)
            ]

            out_df         = df.copy()
            out_df["text"] = transcriptions
            out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            tqdm.write(f"      Saved: {os.path.basename(out_csv)}")

    print("\n  Done.")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Text → TTS audio → noisy audio → ASR text"
    )
    parser.add_argument(
        "--step", choices=["tts", "mix", "asr", "all"], default="all",
        help="Stage to run (default: all)"
    )
    parser.add_argument(
        "--noise-type", nargs="+", choices=NOISE_TYPES, default=None,
        metavar="TYPE",
        help="Noise type(s) for mix/asr (default: all)"
    )
    parser.add_argument(
        "--snr", nargs="+", type=int, default=None,
        metavar="DB",
        help="SNR levels in dB for mix/asr (default: all)"
    )
    args = parser.parse_args()

    snr = args.snr or SNR_LEVELS

    if args.step in ("tts", "all"):
        stage_tts()
    if args.step in ("mix", "all"):
        stage_mix(noise_types=args.noise_type, snr_levels=snr)
    if args.step in ("asr", "all"):
        stage_asr(noise_types=args.noise_type, snr_levels=snr)


if __name__ == "__main__":
    main()
