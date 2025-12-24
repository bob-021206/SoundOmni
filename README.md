# SoundOmni

SoundOmni is a unified multimodal foundation model for audio generation and understanding, built upon the BAGEL (Bidirectional Audio Generation and Understanding with Expert Learning) architecture. The model integrates text-to-audio generation capabilities with audio understanding, leveraging a Mixture-of-Tokens (MoT) expert routing mechanism to efficiently handle both understanding and generation tasks.

## Overview

SoundOmni extends the BAGEL framework to support audio modalities, enabling:
- **Text-to-Audio Generation**: Generate high-quality audio from textual descriptions using flow matching
- **Audio Understanding**: Process and understand audio content through a dedicated understanding branch
- **Unified Training**: Joint training of both understanding and generation capabilities

The architecture employs separate expert pathways for understanding and generation tasks, allowing the model to efficiently route tokens through specialized components while sharing a common language model backbone.

## Key Features

- **Bidirectional Audio Processing**: Simultaneous support for audio understanding and generation
- **Mixture-of-Tokens Architecture**: Expert routing mechanism that directs tokens to specialized understanding or generation pathways
- **Stable Audio VAE Integration**: Uses Stable Audio VAE for efficient audio latent space representation
- **Flow Matching**: Implements flow matching for high-quality audio generation
- **Distributed Training**: Full support for FSDP (Fully Sharded Data Parallel) training
- **Packed Sequence Training**: Efficient sequence packing for variable-length audio-text pairs

## Architecture

The model architecture consists of:

1. **Language Model Backbone**: Based on Qwen2-Audio-7B-Instruct, providing the core language understanding capabilities
2. **Understanding Branch**: Processes input audio and text through dedicated experts
3. **Generation Branch**: Generates audio latents from text prompts using specialized generation experts
4. **Audio VAE**: Stable Audio VAE for encoding/decoding between audio waveforms and latent representations
5. **Connectors**: MLP connectors that bridge between audio latent space and language model embeddings

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- CUDA-capable GPU(s) for training and inference

### Dependencies

Install the required dependencies:

```bash
pip install torch torchaudio transformers
pip install stable-audio-tools
pip install wandb soundfile
pip install safetensors
```

## Quick Start

### Training

To train the model, use the provided training script:

```bash
torchrun --nproc_per_node=4 train/pretrain_unified_navit.py \
    --model_path "hf/BAGEL-7B-MoT" \
    --llm_path "Qwen/Qwen2-Audio-7B-Instruct" \
    --qwen2_audio_path "Qwen/Qwen2-Audio-7B-Instruct" \
    --audio_gen True \
    --audio_und True \
    --gen_intermediate_size 2048 \
    --online_audio_encoding True \
    --audio_vae_path "ckpt/stable_audio_open_vae_weights.pth" \
    --audio_vae_config "configs/audio_vae.json" \
    --dataset_config_file "data/configs/audiocaps_train.yaml" \
    --total_steps 50000 \
    --lr 1e-4
```

### Inference

For standalone inference:

```bash
python run_inference_standalone.py \
    --checkpoint "path/to/checkpoint.safetensors" \
    --llm_path "Qwen/Qwen2-Audio-7B-Instruct" \
    --vae_path "ckpt/stable_audio_open_vae_weights.pth" \
    --vae_config "configs/audio_vae.json" \
    --output_dir "output/" \
    --steps 25 \
    --cfg 1.0
```

## Configuration

### Dataset Configuration

Dataset configurations are specified in YAML files under `data/configs/`. Example configuration for text-to-audio training:

```yaml
t2a_pretrain:
  dataset_names:
  - audiocaps
  audio_transform_args:
    sample_rate: 44100
    max_duration: 10.0
    normalize: true
    num_channels: 2
  is_mandatory: true
  num_used_data:
  - 100
  weight: 1
  split: train
```

### Model Configuration

Model hyperparameters can be configured through command-line arguments or configuration files. Key parameters include:

- `--audio_gen`: Enable audio generation branch
- `--audio_und`: Enable audio understanding branch
- `--gen_intermediate_size`: Intermediate size for generation expert bottleneck
- `--timestep_shift`: Flow matching timestep shift parameter
- `--audio_vae_path`: Path to Stable Audio VAE checkpoint
- `--audio_vae_config`: Path to VAE configuration JSON

## Project Structure

```
SoundOmni/
├── configs/              # Model and VAE configuration files
├── data/                 # Dataset loading and preprocessing
│   ├── configs/          # Dataset configuration YAML files
│   ├── t2a_dataset.py   # Text-to-audio dataset implementation
│   └── ...
├── models/               # Model implementations
│   ├── bagel.py         # Main BAGEL model architecture
│   ├── audio_vae.py     # Audio VAE wrapper
│   └── layers/          # Custom layer implementations
├── train/                # Training scripts and utilities
│   ├── pretrain_unified_navit.py  # Main training script
│   └── fsdp_utils.py    # FSDP training utilities
├── run_inference_standalone.py    # Standalone inference script
└── scripts/              # Training shell scripts
```

## Training Details

### Distributed Training

The codebase supports distributed training using PyTorch's FSDP (Fully Sharded Data Parallel). Key features:

- Automatic sharding of model parameters across GPUs
- Gradient checkpointing for memory efficiency
- EMA (Exponential Moving Average) model tracking
- Automatic checkpoint resumption

### Training Features

- **Packed Sequences**: Efficient batching of variable-length sequences
- **Online Audio Encoding**: On-the-fly encoding of audio to latents during training
- **Gradient Accumulation**: Support for gradient accumulation to simulate larger batch sizes
- **Learning Rate Scheduling**: Cosine or constant learning rate schedules with warmup
- **Loss Components**: Combined cross-entropy (text) and MSE (audio) losses

## Inference

The inference pipeline supports:

- **Text-to-Audio Generation**: Generate audio from text prompts
- **Configurable Sampling**: Adjustable number of diffusion steps and CFG scale
- **Batch Processing**: Efficient batch generation for multiple prompts
- **Trainset Evaluation**: Evaluate on training set examples with reference audio

## Citation

If you use SoundOmni in your research, please cite:

```bibtex
@software{soundomni2025,
  title={SoundOmni: Unified Audio Generation and Understanding},
  author={Bytedance},
  year={2025},
  url={https://github.com/your-repo/soundomni}
}
```

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Acknowledgments

- Built upon [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) and the BAGEL architecture
- Uses [Stable Audio](https://github.com/Stability-AI/stable-audio-tools) VAE for audio encoding/decoding
- Training infrastructure inspired by modern large-scale language model training practices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and issues, please open an issue on the GitHub repository.

