"""
Generate synthetic noise files for the audio noise pipeline.

Outputs (Dataset/noise/):
  white_noise.wav   — Gaussian white noise
  pink_noise.wav    — 1/f pink noise  (proxy for cafe background)
  cafe_noise.wav    — modulated pink noise  (speech-like babble spectrum)
  street_noise.wav  — low-frequency shaped noise  (traffic rumble)
  crowd_noise.wav   — mid-band modulated noise  (crowd chatter)
  traffic_noise.wav — very low-frequency dominant + random bursts

All files: mono, 16 kHz, 60 seconds, normalised to -3 dBFS.

Usage:
    python download_noise.py

Replace any of these with real recordings (same filename) for more
realistic experiments.  A good free source is the DEMAND database:
  https://zenodo.org/record/1227121
"""

import os
import numpy as np
import torchaudio
import torch

OUTPUT_DIR = "./Dataset/noise"
DURATION_S = 60
SR         = 16_000
SEED       = 42


def _save(name: str, samples: np.ndarray):
    samples = samples / (np.abs(samples).max() + 1e-10) * 0.708  # -3 dBFS
    wav = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    path = os.path.join(OUTPUT_DIR, f"{name}_noise.wav")
    torchaudio.save(path, wav, SR)
    print(f"  Saved {path}")


def _white(n: int, rng) -> np.ndarray:
    return rng.standard_normal(n)


def _pink(n: int, rng) -> np.ndarray:
    """1/f pink noise via spectral shaping."""
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    spectrum[0] = 0.0
    return np.real(np.fft.irfft(spectrum, n))


def _shaped(n: int, rng, low_hz: float, high_hz: float, rolloff: float = 2.0) -> np.ndarray:
    """Band-shaped noise: boost [low_hz, high_hz], roll off outside."""
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    spectrum = np.fft.rfft(white)
    gain = np.ones(len(freqs))
    gain[freqs < low_hz]  = (freqs[freqs < low_hz]  / max(low_hz,  1)) ** rolloff
    gain[freqs > high_hz] = (high_hz / freqs[freqs > high_hz]) ** rolloff
    return np.real(np.fft.irfft(spectrum * gain, n))


def _am_modulate(signal: np.ndarray, rate_hz: float, depth: float, rng) -> np.ndarray:
    """Amplitude-modulate signal with a slow sinusoid + jitter (mimics babble)."""
    t = np.arange(len(signal)) / SR
    mod = 1.0 + depth * np.sin(2 * np.pi * rate_hz * t + rng.uniform(0, 2 * np.pi))
    return signal * mod


def _add_bursts(signal: np.ndarray, n_bursts: int, burst_len_s: float, rng) -> np.ndarray:
    """Add random amplitude bursts (car horns / events)."""
    out = signal.copy()
    burst_samples = int(burst_len_s * SR)
    for _ in range(n_bursts):
        start = rng.integers(0, max(1, len(signal) - burst_samples))
        out[start:start + burst_samples] += rng.standard_normal(burst_samples) * 3.0
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n   = DURATION_S * SR

    print(f"Generating noise files in {OUTPUT_DIR}/ ...")

    # White noise
    _save("white", _white(n, rng))

    # Pink (1/f) noise
    _save("pink", _pink(n, rng))

    # Cafe: pink base + AM modulation at ~3 Hz (speech rhythm) + mid-band boost
    cafe = _shaped(n, rng, low_hz=200, high_hz=3500, rolloff=1.5)
    cafe = _am_modulate(cafe, rate_hz=2.8, depth=0.4, rng=rng)
    _save("cafe", cafe)

    # Street: very low frequency (<400 Hz) dominant rumble
    street = _shaped(n, rng, low_hz=20, high_hz=400, rolloff=3.0)
    street = _add_bursts(street, n_bursts=20, burst_len_s=0.3, rng=rng)
    _save("street", street)

    # Crowd: mid-band (300–3000 Hz) modulated babble
    crowd = _shaped(n, rng, low_hz=300, high_hz=3000, rolloff=2.0)
    crowd = _am_modulate(crowd, rate_hz=4.5, depth=0.6, rng=rng)
    _save("crowd", crowd)

    # Traffic: low-frequency dominant + occasional horn bursts
    traffic = _shaped(n, rng, low_hz=20, high_hz=250, rolloff=4.0)
    traffic = _add_bursts(traffic, n_bursts=8, burst_len_s=0.5, rng=rng)
    _save("traffic", traffic)

    print(f"\nDone. 6 noise files written to {OUTPUT_DIR}/")
    print("Replace any file with a real recording of the same name for better realism.")
    print("Recommended source: DEMAND database  https://zenodo.org/record/1227121")


if __name__ == "__main__":
    main()
