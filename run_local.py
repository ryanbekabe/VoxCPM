"""Local launcher for this machine (RTX 4050 Laptop 6 GB VRAM, Windows).

Wraps app.py without modifying it:
  - builds VoxCPM weights directly in bf16 (avoids the ~9 GB fp32 CPU peak that
    crashed with "access violation" when system commit memory ran low)
  - runs the SenseVoice ASR (prompt auto-transcription) and the ZipEnhancer
    denoiser on CPU so the TTS model alone fits in 6 GB VRAM (otherwise VRAM
    spills into shared system memory and generation becomes ~10x slower)
  - stores model downloads in E:\\hf_cache (C:/D: are low on space)
  - binds to 127.0.0.1 by default

Usage:  .venv\\Scripts\\python.exe run_local.py [--port 8808] [--host 127.0.0.1]
"""

import os

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("MODELSCOPE_CACHE", r"E:\modelscope_cache")

import argparse

import torch

import app


_orig_init = app.VoxCPMDemo.__init__
_orig_load = app.VoxCPMDemo.get_or_load_voxcpm


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.asr_device = "cpu"


def _load(self):
    if self.voxcpm_model is not None:
        return self.voxcpm_model
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return _orig_load(self)
    finally:
        torch.set_default_dtype(prev)


def _zip_init(self, model_path="iic/speech_zipenhancer_ans_multiloss_16k_base"):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    self.model_path = model_path
    self._pipeline = pipeline(Tasks.acoustic_noise_suppression, model=model_path, device="cpu")


import voxcpm.zipenhancer as _zip

_zip.ZipEnhancer.__init__ = _zip_init

app.VoxCPMDemo.__init__ = _init
app.VoxCPMDemo.get_or_load_voxcpm = _load


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openbmb/VoxCPM2")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    app.run_demo(model_id=args.model_id, server_name=args.host, server_port=args.port, device=args.device)
