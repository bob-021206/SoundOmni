# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
from torchaudio import transforms as T

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

def set_audio_channels(audio, target_channels):
    # Add channel dim if it's missing
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
        
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio

def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio
class AudioTransform:
    def __init__(self, **kwargs):
        self.target_sample_rate = kwargs.get("target_sample_rate", 441000)
        self.target_length = kwargs.get("target_length", 441000)
        self.target_channels = kwargs.get("target_channels", 2)
        self.device = kwargs.get("device", "cpu")

    def __call__(self, audio_path: str) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(audio_path)
        audio = prepare_audio(audio,
                                  in_sr=sample_rate,
                                  target_sr=self.target_sample_rate,
                                  target_length=self.target_length,
                                  target_channels=self.target_channels,
                                  device=self.device,
                                  )
        return audio.squeeze(0)
