"""Contoh Text-to-Speech Bahasa Indonesia dengan VoxCPM2.

Menghasilkan beberapa file WAV di folder output_id/:
  1. Suara default (tanpa referensi)
  2. Voice Design - suara pria dewasa, tenang (ala pembaca berita)
  3. Voice Design - suara perempuan muda, ceria
  4. Voice Cloning - meniru suara dari examples/reference_speaker.wav

Jalankan:
    .venv\\Scripts\\python.exe contoh_tts_indonesia.py
"""

import os

os.environ.setdefault("HF_HOME", r"E:\hf_cache")

import time
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

OUT_DIR = Path(__file__).parent / "output_id"
OUT_DIR.mkdir(exist_ok=True)

SAMPLES = [
    {
        "file": "01_default.wav",
        "text": "Halo, selamat datang. Ini adalah contoh suara buatan kecerdasan buatan "
                "yang berbicara dalam bahasa Indonesia.",
    },
    {
        "file": "02_pembaca_berita.wav",
        # Format Voice Design: "(deskripsi suara)teks"
        "text": "(A middle-aged man, calm and authoritative, like a news anchor)"
                "Selamat malam pemirsa. Hari ini pemerintah mengumumkan rencana "
                "pembangunan jalur kereta cepat yang akan menghubungkan Jakarta dan Surabaya.",
    },
    {
        "file": "03_perempuan_ceria.wav",
        "text": "(A young woman, cheerful and energetic, speaking slightly fast)"
                "Wah, akhirnya akhir pekan tiba! Yuk kita jalan-jalan ke pantai, "
                "makan seafood, terus nonton matahari terbenam bareng-bareng.",
    },
    {
        "file": "04_kloning_suara.wav",
        "text": "Teknologi kloning suara memungkinkan kita meniru warna suara seseorang "
                "hanya dari potongan rekaman yang singkat.",
        "reference_wav_path": str(Path(__file__).parent / "examples" / "reference_speaker.wav"),
    },
]


def main():
    t0 = time.time()
    # Bangun bobot langsung dalam bf16 agar tidak kehabisan RAM (lihat run_local.py)
    torch.set_default_dtype(torch.bfloat16)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    torch.set_default_dtype(torch.float32)
    sr = model.tts_model.sample_rate
    print(f"Model dimuat dalam {time.time() - t0:.1f} detik\n")

    for s in SAMPLES:
        t = time.time()
        wav = model.generate(
            text=s["text"],
            reference_wav_path=s.get("reference_wav_path"),
            cfg_value=2.0,
            inference_timesteps=10,
            seed=42,
        )
        out = OUT_DIR / s["file"]
        sf.write(out, wav, sr)
        print(f"[OK] {out.name}: {len(wav) / sr:.1f} detik audio, dibuat dalam {time.time() - t:.1f} detik")

    print(f"\nSemua file tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
