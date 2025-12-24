# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import sys
from contextlib import nullcontext
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb
import yaml
import soundfile as sf
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from models.audio_vae import VAEWrapper
from models.bagel import BagelConfig, Bagel
from modeling.bagel import (
    Qwen2Config, Qwen2ForCausalLM
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


def run_inference_test(
    model,
    tokenizer,
    vae_wrapper,
    new_token_ids,
    step,
    output_dir,
    device,
    logger,
    data=None,
    *,
    num_prompts: int = 1,
    audio_len: int = 64,
    num_timesteps: int = 8,
):
    logger.info(f"Running inference at step {step}...")
    
    prompts = []
    if data is not None and 'packed_text_ids' in data:
        try:
            # Decode text
            full_text = tokenizer.decode(data['packed_text_ids'].cpu().tolist(), skip_special_tokens=False)
            logger.info(f"Decoded batch text sample (first 500 chars): {full_text[:500]}")
            
            # Split by audio start token
            # We know the structure is roughly: [Text] <|im_end|> <|audio_bos|> ...
            
            audio_start_token = None
            if '<|audio_bos|>' in full_text:
                audio_start_token = '<|audio_bos|>'
            elif '<|audio_start|>' in full_text:
                audio_start_token = '<|audio_start|>'
            
            if audio_start_token:
                parts = full_text.split(audio_start_token)
                # parts[0] is text before first audio
                # parts[1] is audio + text before second audio...
                
                for part in parts[:-1]: # The last part is just the end of the sequence
                    # part ends with <|im_end|> usually.
                    # We want the text between the last <|im_start|> and <|im_end|>
                    if '<|im_end|>' in part:
                        # Take content before <|im_end|>
                        pre_end = part.rsplit('<|im_end|>', 1)[0]
                        # Find last <|im_start|>
                        if '<|im_start|>' in pre_end:
                            p = pre_end.rsplit('<|im_start|>', 1)[-1]
                            p = p.strip()
                            if p:
                                prompts.append(p)
            
            # Limit prompts for online inference
            prompts = prompts[: max(1, int(num_prompts))]
            logger.info(f"Extracted {len(prompts)} prompts from training batch.")
        except Exception as e:
            logger.warning(f"Failed to extract prompts from batch: {e}")

    if not prompts:
        logger.info("Using default prompts.")
        prompts = [
            "a small dog whimpering and howling",
            "birds chirp as a vehicle passes by",
            "Sirens wailing in the distance",
            "Birds chirping in a forest"
        ]
        prompts = prompts[: max(1, int(num_prompts))]
    
    logger.info(
        f"Inference Prompts (num_prompts={max(1, int(num_prompts))}, audio_len={int(audio_len)}, num_timesteps={int(num_timesteps)}): {prompts}"
    )
    
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    # Disable compilation for inference to avoid recompilation overhead
    # and potential distributed issues
    torch._dynamo.reset()

    # FSDP note:
    # - Under FSDP flattening, embedding weights become 1D FlatParameters and torch.embedding fails
    #   if we call custom methods that bypass FSDP.forward.
    # - Calling into FSDP-wrapped submodules from rank0 only can deadlock because FSDP uses
    #   collectives (all-gather) during forward.
    # Strategy:
    #   1) All ranks enter summon_full_params(rank0_only=True)
    #   2) Only rank0 runs inference on the *unwrapped* module (model.module)
    #      so no FSDP collectives are used during the forward.
    if isinstance(model, FSDP):
        fsdp_wrapper = model
        raw_model = model.module
        ctx = FSDP.summon_full_params(fsdp_wrapper, recurse=True, writeback=False, rank0_only=True)
    else:
        fsdp_wrapper = None
        raw_model = model
        ctx = nullcontext()

    was_training = raw_model.training
    raw_model.eval()

    try:
        with ctx:
            # Only rank0 executes the actual forward/generation.
            if rank == 0:
                # Avoid rare flash-attn varlen hangs during online inference by forcing
                # PackedAttention to use the PyTorch SDPA fallback for this test only.
                _flash_prev = None
                _sdp_prev = None
                try:
                    from models.layers import packed_attention as _packed_attn_mod
                    _flash_prev = getattr(_packed_attn_mod, "FLASH_ATTN_AVAILABLE", None)
                    if _flash_prev is not None:
                        _packed_attn_mod.FLASH_ATTN_AVAILABLE = False
                except Exception as _e:
                    logger.warning(f"Could not toggle flash-attn for inference test: {_e}")

                # Also force PyTorch to use the math SDP kernel (disables flash/mem-efficient)
                # to avoid occasional kernel-level stalls on some setups.
                try:
                    import torch.backends.cuda as _cuda_backends
                    _sdp_prev = (
                        _cuda_backends.flash_sdp_enabled(),
                        _cuda_backends.mem_efficient_sdp_enabled(),
                        _cuda_backends.math_sdp_enabled(),
                    )
                    _cuda_backends.enable_flash_sdp(False)
                    _cuda_backends.enable_mem_efficient_sdp(False)
                    _cuda_backends.enable_math_sdp(True)
                except Exception as _e:
                    logger.warning(f"Could not toggle torch SDP backends for inference test: {_e}")

                with torch.no_grad():
                    B = len(prompts)
                    curr_kvlens = [0] * B
                    curr_rope = [0] * B

                    # 1. Prepare Prompts
                    logger.info("Step 1: Prepare Prompts")
                    generation_input, newlens, new_rope = raw_model.prepare_prompts(
                        curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids
                    )

                    # Move to device
                    for k, v in generation_input.items():
                        if isinstance(v, torch.Tensor):
                            generation_input[k] = v.to(device)

                    # 2. Prefill Text
                    logger.info("Step 2: Prefill Text (forward_cache_update_text)")
                    _t0 = time()
                    past_key_values = raw_model.forward_cache_update_text(
                        past_key_values=None,
                        **generation_input
                    )
                    torch.cuda.synchronize()
                    logger.info(f"Step 2: Prefill Text Done (elapsed={time() - _t0:.2f}s)")

                    # 3. Prepare Audio Inputs
                    # Online inference is a smoke test; keep it small to avoid stalling training.
                    audio_len = int(audio_len)
                    audio_lengths = [audio_len] * B

                    logger.info("Step 3: Prepare Audio Inputs")
                    gen_input_audio = raw_model.prepare_audio_latents(
                        newlens, new_rope, audio_lengths, new_token_ids
                    )

                    for k, v in gen_input_audio.items():
                        if isinstance(v, torch.Tensor):
                            gen_input_audio[k] = v.to(device)

                    # 4. Generate
                    logger.info("Step 4: Generate Audio (model.generate_audio)")
                    _t0 = time()
                    latents_list = raw_model.generate_audio(
                        past_key_values=past_key_values,
                        num_timesteps=int(num_timesteps),
                        **gen_input_audio
                    )
                    torch.cuda.synchronize()
                    logger.info(f"Step 4: Generate Audio Done (elapsed={time() - _t0:.2f}s)")

                    # 5. Decode and Save (rank0 only)
                    save_dir = os.path.join(output_dir, f"inference_step_{step}")
                    os.makedirs(save_dir, exist_ok=True)

                    for i, latents in enumerate(latents_list):
                        # latents: [L, D] -> [1, D, L]
                        latents = latents.unsqueeze(0).permute(0, 2, 1)
                        audio = vae_wrapper.model.decode_audio(latents, chunk_size=vae_wrapper.chunk_size)
                        audio = audio.squeeze().cpu().numpy()

                        # If stereo returned as [C, T], transpose to [T, C] for soundfile.
                        if len(audio.shape) == 2 and audio.shape[0] < audio.shape[1]:
                            audio = audio.T

                        path = os.path.join(save_dir, f"sample_{i}.wav")
                        sf.write(path, audio, int(vae_wrapper.target_sample_rate))

                        txt_path = os.path.join(save_dir, f"sample_{i}.txt")
                        with open(txt_path, "w") as f:
                            f.write(prompts[i])

                    logger.info(f"Inference done. Saved to {save_dir}")

                # Restore flash-attn flag
                try:
                    if _flash_prev is not None:
                        from models.layers import packed_attention as _packed_attn_mod
                        _packed_attn_mod.FLASH_ATTN_AVAILABLE = _flash_prev
                except Exception:
                    pass

                # Restore SDP backend flags
                try:
                    if _sdp_prev is not None:
                        import torch.backends.cuda as _cuda_backends
                        _cuda_backends.enable_flash_sdp(_sdp_prev[0])
                        _cuda_backends.enable_mem_efficient_sdp(_sdp_prev[1])
                        _cuda_backends.enable_math_sdp(_sdp_prev[2])
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

    raw_model.train(was_training)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    qwen2_audio_path: str = field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        metadata={"help": "Path to Qwen2-Audio model for initializing audio understanding branch."}
    )
    gen_intermediate_size: int = field(
        default=None,
        metadata={"help": "Intermediate size for the Generation Expert (Bottleneck). If None, uses same as Understanding Expert."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    audio_gen: bool = field(
        default=False,
        metadata={"help": "Train audio generation branch."}
    )
    audio_und: bool = field(
        default=False,
        metadata={"help": "Train audio understanding branch."}
    )
    audio_vae_path: str = field(
        default=None,
        metadata={"help": "Path to Audio VAE checkpoint."}
    )
    audio_vae_config: str = field(
        default=None,
        metadata={"help": "Path to Audio VAE config json."}
    )
    online_audio_encoding: bool = field(
        default=True,
        metadata={"help": "Encode audio to latent on-the-fly."}
    )
    freeze_audio_vae: bool = field(
        default=True,
        metadata={"help": "Freeze Audio VAE."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    eval_every: int = field(
        default=2000,
        metadata={"help": "Run inference test every N training steps."}
    )

    enable_inference_test: bool = field(
        default=False,
        metadata={"help": "If True, run a small text-to-audio inference and save wavs every eval_every steps (rank0 saves)."}
    )

    inference_test_num_prompts: int = field(
        default=1,
        metadata={"help": "Number of prompts to run during online inference test (keep small to avoid stalling training)."}
    )
    inference_test_audio_len: int = field(
        default=64,
        metadata={"help": "Number of audio latent tokens to generate during online inference test (keep small)."}
    )
    inference_test_num_timesteps: int = field(
        default=8,
        metadata={"help": "Number of diffusion steps for online inference test (keep small)."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = False # Visual und removed
    if model_args.gen_intermediate_size is not None:
        llm_config.gen_intermediate_size = model_args.gen_intermediate_size
    
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    # Setup Audio VAE
    audio_vae_wrapper = None
    if training_args.audio_gen:
        if training_args.audio_vae_path and training_args.audio_vae_config:
             audio_vae_wrapper = VAEWrapper(
                model_config_path=training_args.audio_vae_config,
                model_ckpt_path=training_args.audio_vae_path,
                device=torch.device("cuda", device)
            )
             if training_args.freeze_audio_vae:
                for p in audio_vae_wrapper.model.parameters():
                    p.requires_grad = False
        else:
             logger.warning("Audio Gen enabled but VAE path/config not provided. Assuming simulated latents or pre-encoded data.")

    config = BagelConfig(
        visual_gen=False,
        visual_und=False,
        audio_gen=training_args.audio_gen,
        audio_und=training_args.audio_und,
        llm_config=llm_config, 
        vit_config=None,
        vae_config=None,
        latent_patch_size=2, # Default
        max_latent_size=32, # Default
        vit_max_num_patch_per_side=70, # Default
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        None, 
        config
    )

    # Load Qwen2-Audio Weights
    if training_args.audio_und:
        logger.info(f"Loading Qwen2-Audio weights from {model_args.qwen2_audio_path}")
        model = Bagel.audio_from_pretrained_qwen2_audio(
            model, 
            model_args.qwen2_audio_path,
            device_map="cpu"
        )
        
        # Freeze Audio Und Branch
        for p in model.audio_encoder.parameters(): p.requires_grad = False
        for p in model.audio_projector.parameters(): p.requires_grad = False

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    # Freeze LLM Base (Understanding Branch)
    # We only want to train the "Generation" components:
    # 1. *_moe_gen layers (Attention and MLP experts for generation)
    # 2. Audio VAE adapters (audiovae2llm, llm2audiovae)
    # 3. Time embedder and Position embeddings
    
    logger.info("Freezing Understanding Branch...")
    for name, param in model.named_parameters():
        # Default to frozen
        param.requires_grad = False
        
        # Enable gradients for Generation components
        if "moe_gen" in name:
            param.requires_grad = True
        elif "audiovae2llm" in name or "llm2audiovae" in name:
            param.requires_grad = True
        elif "time_embedder" in name:
            param.requires_grad = True
        elif "gen_pos_embed" in name:
            param.requires_grad = True
            
    # Ensure uniform dtype (bfloat16) for FSDP and move to device
    model = model.to(device=device, dtype=torch.bfloat16)

    # Debug: Print trainable parameters
    if dist.get_rank() == 0:
        logger.info("Trainable Parameters Inspection:")
        trainable_params = 0
        all_params = 0
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # logger.info(f"  Trainable: {name} | Shape: {param.shape} | Dtype: {param.dtype}")
        
        logger.info(f"Total Params: {all_params/1e9:.2f}B")
        logger.info(f"Trainable Params: {trainable_params/1e9:.2f}B")
        logger.info(f"Trainable Ratio: {trainable_params/all_params*100:.2f}%")

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )
        # Force update LR from arguments if resuming
        # Because optimizer.load_state_dict overwrites the LR with the checkpoint value
        for param_group in optimizer.param_groups:
            param_group['lr'] = training_args.lr
        logger.info(f"Forced optimizer LR to {training_args.lr}")

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    
    # Ensure special tokens are added to tokenizer before creating dataset
    # (Assuming tokenizer is already set up before this block)
    
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # Prepare models for training:
    fsdp_model.train()
    ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        
        # Handle dictionary data from DummyDataset
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v.to(device)
                else:
                    new_data[k] = v
            data = new_data
        else:
            data = data.cuda(device).to_dict()
            
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)       
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.audio_gen:
                # --- Audio Data Processing ---
                if 'padded_audio_features' in data and training_args.online_audio_encoding and audio_vae_wrapper:
                    # 1. Get raw audio from batch
                    audio_waveforms = data['padded_audio_features'] # [B, C, T]
                    
                    # 2. Encode with VAE
                    with torch.no_grad():
                        # Encode: [B, C, T] -> [B, D, L]
                        latents = audio_vae_wrapper.model.encode_audio(audio_waveforms, chunk_size=audio_vae_wrapper.chunk_size)
                        # Permute to [B, L, D]
                        latents = latents.permute(0, 2, 1)
                        
                    # 3. Flatten latents for packed sequence
                    # We need to extract the valid tokens for each sample and concatenate them
                    # We use audio_feature_shapes to determine valid length
                    valid_latents = []
                    if 'audio_feature_shapes' in data:
                        for i, shape in enumerate(data['audio_feature_shapes']):
                            T_raw = shape[1]
                            L_latent = T_raw // 2048
                            if L_latent == 0: L_latent = 1
                            valid_latents.append(latents[i, :L_latent, :])
                    else:
                        # Fallback if shapes not available (should not happen with PackedDataset)
                        valid_latents.append(latents.reshape(-1, latents.shape[-1]))

                    data['packed_audio_latents'] = torch.cat(valid_latents, dim=0)

                    # Debug: Print Latent Statistics
                    if micro_step % 10 == 0 and dist.get_rank() == 0:
                        latents_mean = data['packed_audio_latents'].mean().item()
                        latents_std = data['packed_audio_latents'].std().item()
                        latents_min = data['packed_audio_latents'].min().item()
                        latents_max = data['packed_audio_latents'].max().item()
                        logger.info(f"Step {curr_step} Latent Stats: Mean={latents_mean:.4f}, Std={latents_std:.4f}, Min={latents_min:.4f}, Max={latents_max:.4f}")

                    # Remove raw features to save memory
                    del data['padded_audio_features']
                    if 'audio_feature_shapes' in data:
                        del data['audio_feature_shapes']
            
            if micro_step == 0:
                logger.info(f"--- Debug Info at Step {curr_step} ---")
                logger.info(f"Sample Lens: {data.get('sample_lens', 'N/A')}")
                if 'packed_text_ids' in data:
                    logger.info(f"Packed Text IDs Shape: {data['packed_text_ids'].shape}")
                    logger.info(f"Packed Text IDs (first 20): {data['packed_text_ids'][:20]}")
                if 'packed_audio_latents' in data:
                    logger.info(f"Packed Audio Latents Shape: {data['packed_audio_latents'].shape}")
                    logger.info(f"Packed Audio Latents (first 5 rows): {data['packed_audio_latents'][:5]}")
                if 'packed_audio_gen_token_indexes' in data:
                    logger.info(f"Packed Audio Gen Token Indexes Shape: {data['packed_audio_gen_token_indexes'].shape}")
                    logger.info(f"Packed Audio Gen Token Indexes (first 20): {data['packed_audio_gen_token_indexes'][:20]}")
                logger.info("-------------------------------------")

            try:
                loss_dict = fsdp_model(**data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        loss = 0
        total_mse_tokens = torch.tensor(0, device=device)
        total_audio_mse_tokens = torch.tensor(0, device=device)
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.audio_gen:
            # Audio MSE Loss
            audio_mse = loss_dict["audio_mse"]
            if audio_mse is not None:
                total_audio_mse_tokens = torch.tensor(len(data['audio_mse_loss_indexes']), device=device)
                dist.all_reduce(total_audio_mse_tokens, op=dist.ReduceOp.SUM)
                # audio_mse is [N, D] or [N]? Check Bagel forward.
                # Bagel forward: audio_mse = (pred - target)**2. Shape [N_active, D].
                audio_mse = audio_mse.mean(dim=-1).sum() * dist.get_world_size() / total_audio_mse_tokens
                loss_dict["audio_mse"] = audio_mse.detach()
                loss = loss + audio_mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()
        
        # Debug: Print Grad Norm
        if micro_step % 10 == 0 and dist.get_rank() == 0:
            total_norm_debug = 0.0
            for p in fsdp_model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm_debug += param_norm.item() ** 2
            total_norm_debug = total_norm_debug ** 0.5
            logger.info(f"Step {curr_step} Grad Norm: {total_norm_debug:.4f}")

        fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
        optimizer.zero_grad()
        
        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                if value is None:
                    continue
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if training_args.enable_inference_test and curr_step > 0 and curr_step % training_args.eval_every == 0:
            # Safer than summon_full_params(rank0_only=True): run the forward on all ranks,
            # but only rank0 writes wavs. This avoids FSDP gather/reshard deadlocks.
            dist.barrier()
            try:
                run_inference_test(
                    model=ema_model if ema_model is not None else fsdp_model,
                    tokenizer=tokenizer,
                    vae_wrapper=audio_vae_wrapper,
                    new_token_ids=new_token_ids,
                    step=curr_step,
                    output_dir=training_args.results_dir,
                    device=device,
                    logger=logger,
                    data=data,
                    num_prompts=training_args.inference_test_num_prompts,
                    audio_len=training_args.inference_test_audio_len,
                    num_timesteps=training_args.inference_test_num_timesteps,
                )
            except Exception as e:
                logger.error(f"Error during inference test at step {curr_step}: {e}")
            dist.barrier()
            fsdp_model.train()

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # Clear caches and ensure all CUDA operations complete before checkpoint
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            # Clear CUDA cache and force garbage collection after checkpoint to free memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
    
    # Save final checkpoint if not already saved
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        # Clear caches and ensure all CUDA operations complete before final checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
        
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=curr_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list
        )
        # Clear CUDA cache and force garbage collection after final checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
