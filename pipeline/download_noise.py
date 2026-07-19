"""
Download real-world noise files for the audio noise pipeline.

Noise types:
  white   → synthetic Gaussian baseline
  babble  → MUSAN noise subset (real multi-speaker babble)
             OpenSLR 17: https://www.openslr.org/17/
  street  → DEMAND STRAFFIC (real street recording)
             Zenodo 1227121: https://zenodo.org/record/1227121

All outputs: mono, 16 kHz, 60 s, normalised to -3 dBFS → Dataset/noise/

Usage:
    python download_noise.py
"""

import io
import os
import tarfile
import zipfile
from math import gcd

import numpy as np
import requests
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
from scipy.signal import resample_poly

OUTPUT_DIR = "./Dataset/noise"
TARGET_SR  = 16_000
DURATION_S = 300   # 5 minutes — enough for random-offset sampling
SEED       = 42

MUSAN_URL    = "https://www.openslr.org/resources/17/musan.tar.gz"
DEMAND_URLS  = {
    "cafe":  "https://zenodo.org/record/1227121/files/CAFE_16k.zip",      # cafeteria
    "crowd": "https://zenodo.org/record/1227121/files/SPSQUARE_16k.zip",  # city square / crowd
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / (np.abs(x).max() + 1e-10) * 0.708).astype(np.float32)


def _to_float32(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.astype(np.float32)


def _to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim > 1 else data


def _resample(data: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return data
    g = gcd(src_sr, TARGET_SR)
    return resample_poly(data, TARGET_SR // g, src_sr // g).astype(np.float32)


def _trim_pad(data: np.ndarray, n_samples: int) -> np.ndarray:
    if len(data) >= n_samples:
        return data[:n_samples]
    return np.tile(data, int(np.ceil(n_samples / len(data))))[:n_samples]


def _process(data: np.ndarray, sr: int) -> np.ndarray:
    data = _to_float32(data)
    data = _to_mono(data)
    data = _resample(data, sr)
    data = _trim_pad(data, DURATION_S * TARGET_SR)
    return _normalise(data)


def _save(name: str, samples: np.ndarray):
    path = os.path.join(OUTPUT_DIR, f"{name}_noise.wav")
    wav_write(path, TARGET_SR, samples)
    print(f"  -> {path}")


def _get(url: str, desc: str) -> bytes:
    print(f"  Downloading {desc} ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    buf = io.BytesIO()
    done = 0
    for chunk in r.iter_content(1 << 20):
        buf.write(chunk)
        done += len(chunk)
        if total:
            print(f"\r    {done/1e6:.0f} / {total/1e6:.0f} MB", end="", flush=True)
    print()
    return buf.getvalue()


# ------------------------------------------------------------------------------
# Noise generators
# ------------------------------------------------------------------------------

def make_white():
    print("Generating white noise (synthetic) ...")
    rng = np.random.default_rng(SEED)
    data = rng.standard_normal(DURATION_S * TARGET_SR).astype(np.float32)
    _save("white", _normalise(data))


def make_babble(n_speakers: int = 15):
    """
    Stream MUSAN tarball, collect speech clips from diverse speakers,
    overlay them to form babble noise.
    Stops as soon as total collected audio >= n_speakers * DURATION_S.
    """
    out = os.path.join(OUTPUT_DIR, "babble_noise.wav")
    if os.path.exists(out):
        print("  babble_noise.wav already exists — skipping.")
        return

    target_samples = DURATION_S * TARGET_SR
    need_total     = n_speakers * target_samples   # e.g. 15 × 300s

    print(f"Downloading MUSAN (streaming) — need {n_speakers} speakers × {DURATION_S}s ...")

    all_clips: list[np.ndarray] = []
    seen_speakers: set[str]    = set()
    collected = 0

    r = requests.get(MUSAN_URL, stream=True, timeout=600)
    r.raise_for_status()

    with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = member.name.replace("\\", "/").split("/")
            # musan/speech/<source>/<spk_id>/<file>.wav
            if len(parts) < 3 or parts[1] != "speech":
                continue
            if not member.name.lower().endswith(".wav"):
                continue

            spk_id = parts[3] if len(parts) >= 4 else parts[2]
            seen_speakers.add(spk_id)

            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                sr, data = wav_read(io.BytesIO(f.read()))
            except Exception:
                continue

            clip = _to_float32(data)
            clip = _to_mono(clip)
            clip = _resample(clip, sr)
            all_clips.append(clip)
            collected += len(clip)

            print(f"\r    speakers: {len(seen_speakers)}  "
                  f"collected: {collected/TARGET_SR:.0f}s / {need_total/TARGET_SR:.0f}s needed",
                  end="", flush=True)

            if collected >= need_total:
                break

    print()
    if not all_clips:
        raise RuntimeError("No speech clips extracted from MUSAN tarball.")

    # shuffle clips, split into n_speakers equal tracks, then mix
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_clips)

    chunk = max(1, len(all_clips) // n_speakers)
    tracks = []
    for i in range(n_speakers):
        group = all_clips[i * chunk: (i + 1) * chunk]
        if not group:
            continue
        track = _trim_pad(np.concatenate(group), target_samples)
        track = track / (np.abs(track).max() + 1e-10)
        tracks.append(track)

    babble = np.mean(tracks, axis=0).astype(np.float32)
    _save("babble", _normalise(babble))


def make_demand(name: str, url: str):
    out = os.path.join(OUTPUT_DIR, f"{name}_noise.wav")
    if os.path.exists(out):
        print(f"  {name}_noise.wav already exists — skipping.")
        return
    label = {"cafe": "DEMAND CAFE (~200 MB)", "crowd": "DEMAND SPSQUARE city square (~200 MB)"}[name]
    raw = _get(url, label)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        wav_names = [n for n in zf.namelist() if n.lower().endswith(".wav")]
        if not wav_names:
            raise RuntimeError(f"No WAV in {name} zip")
        with zf.open(wav_names[0]) as f:
            sr, data = wav_read(f)
    _save(name, _process(data, sr))


# ------------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}/\n")

    make_white()
    make_babble()
    for name, url in DEMAND_URLS.items():
        make_demand(name, url)

    print("\nAll noise files ready.")
    print("NOISE_TYPES = [\"white\", \"babble\", \"cafe\", \"crowd\"]")


if __name__ == "__main__":
    main()
