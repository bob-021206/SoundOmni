import os
import sys
import torch
import soundfile as sf
import argparse
import yaml
from transformers import AutoTokenizer
from safetensors.torch import load_file
import torchaudio
from tqdm import tqdm
from typing import Optional, Tuple, List

# Add current directory to path
sys.path.append(os.getcwd())

from models.bagel import Bagel, BagelConfig
from models.audio_vae import VAEWrapper
from modeling.bagel import Qwen2ForCausalLM, Qwen2Config
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, PackedDataset

# ================= 配置区域 =================
# [请修改] 你的最新 Checkpoint 路径
CHECKPOINT_PATH = "/mnt/cfs/5vr0p6/yimingjing/workspace/Audio_Text_Project/results/audio_bottleneck_training_test/checkpoints/0003000/model.safetensors" 
# (如果不确定具体是哪个step，请修改上面的路径到具体的 .safetensors 文件)

LLM_PATH = "/mnt/cfs/5vr0p6/yimingjing/workspace/Models/Qwen2-Audio-7B-Instruct"
VAE_PATH = "ckpt/stable_audio_open_vae_weights.pth"
VAE_CONFIG = "configs/audio_vae.json"
OUTPUT_DIR = "result_inference/7k" # 输出目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================


def _save_wav(path: str, audio_tensor: torch.Tensor, sample_rate: int):
    audio = audio_tensor.detach().cpu().to(torch.float32)
    if audio.dim() == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.transpose(0, 1)  # [C, T] -> [T, C]
    sf.write(path, audio.numpy(), int(sample_rate))

@torch.no_grad()
def generate_audio_standalone(
    model,
    packed_text_ids: torch.LongTensor,
    packed_text_indexes: torch.LongTensor,
    packed_init_noises: torch.Tensor,
    packed_audio_gen_position_ids: torch.LongTensor,
    packed_audio_gen_token_indexes: torch.LongTensor,
    packed_seqlens: torch.IntTensor,
    packed_position_ids: torch.LongTensor,
    packed_indexes: torch.LongTensor,
    past_key_values,
    key_values_lens: torch.IntTensor,
    packed_key_value_indexes: torch.LongTensor,
    num_timesteps: int = 20,
    timestep_shift: float = 1.0,
    cfg_scale: float = 1.0, # Renamed for clarity
):
    # Flow Matching Inference: t goes from 1.0 (noise) -> 0.0 (data)
    x_t = packed_init_noises.to(dtype=model.audiovae2llm.weight.dtype)

    # Standard Linear Schedule
    timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
    # Apply shift if needed (usually 1.0 is fine)
    timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
    dts = timesteps[:-1] - timesteps[1:]
    timesteps = timesteps[:-1]

    # Construct Attention Mask
    total_len = packed_seqlens.item()
    indices = torch.arange(total_len, device=x_t.device)
    mask = (indices.unsqueeze(1) >= indices.unsqueeze(0)) 
    attention_mask = torch.zeros((total_len, total_len), dtype=torch.float, device=x_t.device)
    attention_mask = attention_mask.masked_fill(~mask, float("-inf"))
    # Audio Block is Full Attention
    attention_mask[packed_audio_gen_token_indexes.unsqueeze(1), packed_audio_gen_token_indexes.unsqueeze(0)] = 0.0

    # Use train-mode to follow the same forward path as training (Packed* forward_train).
    # Keep dropout disabled for deterministic sampling.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"Sampling (CFG={cfg_scale})", leave=False):
        timestep = torch.full((1,), t, device=x_t.device)
        
        # --- 1. Conditional Prediction (Text Conditioned) ---
        # Input: [Text, Audio]
        # RoPE: Text(0..T-1), Audio(T)
        
        packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((total_len, model.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        hidden_x_t = model.audiovae2llm(x_t)
        time_emb = model.time_embedder(timestep)
        pos_emb = model.gen_pos_embed[packed_audio_gen_position_ids]
        
        audio_gen_input = hidden_x_t + time_emb + pos_emb
        if audio_gen_input.dtype != packed_sequence.dtype:
            audio_gen_input = audio_gen_input.to(packed_sequence.dtype)
        packed_sequence[packed_audio_gen_token_indexes] = audio_gen_input

        # IMPORTANT: call the same entrypoint as training (Qwen2ForCausalLM.forward_train)
        output_sequence = model.language_model.forward_train(
            packed_sequence=packed_sequence,
            sample_lens=[total_len],
            attention_mask=[attention_mask],
            packed_position_ids=packed_position_ids,
            packed_und_token_indexes=packed_text_indexes,
            packed_gen_token_indexes=packed_audio_gen_token_indexes
        )
        
        audio_output = output_sequence[packed_audio_gen_token_indexes]
        v_pred_cond = model.llm2audiovae(audio_output)

        # --- 2. Unconditional Prediction (For CFG) ---
        if cfg_scale != 1.0:
            # Correct Implementation:
            # Input: [Audio] ONLY
            # RoPE: Audio(0) -> Because in training, when text is dropped, curr_rope starts at 0.
            
            # Construct Uncond Input
            # We only need the Audio part.
            # But forward_train expects a packed sequence. 
            # We can create a smaller packed sequence of length A.
            
            A = x_t.shape[0]
            packed_sequence_uncond = audio_gen_input.clone() # [A, D]
            
            # Uncond Attention Mask: Full Attention [A, A]
            attention_mask_uncond = torch.zeros((A, A), dtype=torch.float, device=x_t.device)
            
            # Uncond Position IDs: All 0 (Shared position for image/audio tokens)
            # In dataset_base.py: sequence_status['packed_position_ids'].extend([curr_rope_id] * ...)
            # If text dropped, curr_rope_id = 0.
            packed_position_ids_uncond = torch.zeros(A, dtype=torch.long, device=x_t.device)
            
            # Token Indexes: 0..A-1
            # We need to pass dummy indexes for und/gen to satisfy the forward signature
            # Since we only have Audio (Gen) tokens:
            packed_gen_token_indexes_uncond = torch.arange(A, dtype=torch.long, device=x_t.device)
            packed_und_token_indexes_uncond = torch.tensor([], dtype=torch.long, device=x_t.device) # Empty
            
            output_sequence_uncond = model.language_model.forward_train(
                packed_sequence=packed_sequence_uncond,
                sample_lens=[A],
                attention_mask=[attention_mask_uncond],
                packed_position_ids=packed_position_ids_uncond,
                packed_und_token_indexes=packed_und_token_indexes_uncond,
                packed_gen_token_indexes=packed_gen_token_indexes_uncond
            )
            
            # Output is [A, D]
            audio_output_uncond = output_sequence_uncond
            v_pred_uncond = model.llm2audiovae(audio_output_uncond)
            
            # CFG Formula: v = v_uncond + scale * (v_cond - v_uncond)
            v_t = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
        else:
            v_t = v_pred_cond

        # Euler Step
        x_t = x_t - v_t * dts[i]

    return [x_t]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--llm_path", type=str, default=LLM_PATH)
    parser.add_argument("--vae_path", type=str, default=VAE_PATH)
    parser.add_argument("--vae_config", type=str, default=VAE_CONFIG)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0, help="Fix RNG for init noise; use same seed across checkpoints to compare outputs.")
    parser.add_argument("--prompt", type=str, default=None, help="If set, run only this single prompt instead of the built-in prompt list.")

    # Trainset eval mode
    parser.add_argument(
        "--eval_dataset_yaml",
        type=str,
        default=None,
        help="Dataset YAML (same format as training) to dump ref/recon/sample wavs from trainset examples.",
    )
    parser.add_argument("--eval_num_samples", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determinism: seed controls init noise and any other torch randomness used here.
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    
    # 1. Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = Qwen2Tokenizer.from_pretrained(args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    # 2. Load LLM Config & Model
    print("Loading LLM Config...")
    llm_config = Qwen2Config.from_pretrained(args.llm_path)
    
    # === [CRITICAL FIXES] ===
    # 必须补齐这些配置，否则推理必挂
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True                  # 关键：开启 QK Norm
    llm_config.gen_intermediate_size = 2048    # 关键：Bottleneck 大小
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    # ========================

    llm_config.vocab_size = len(tokenizer)

    print("Loading Qwen2 Base Model...")
    language_model = Qwen2ForCausalLM(llm_config)
    language_model.resize_token_embeddings(len(tokenizer))

    # 3. Initialize Bagel
    print("Initializing Bagel Model...")
    config = BagelConfig(
        visual_gen=False,
        visual_und=False,
        audio_gen=True,
        audio_und=True,
        llm_config=llm_config,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        timestep_shift=1.0,
    )
    
    model = Bagel(language_model, None, config)
    
    # 4. Load Weights
    print(f"Loading weights from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    state_dict = load_file(args.checkpoint)
    
    # Remove 'module.' prefix if present
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # --- DIAGNOSTIC: Check Key Existence and Shapes ---
    print("\n[DIAGNOSTIC] Checking Checkpoint Keys...")
    
    # 1. Check QK Norm
    qk_norm_keys = [k for k in state_dict.keys() if "q_norm" in k]
    if len(qk_norm_keys) > 0:
        print(f"  [INFO] Found {len(qk_norm_keys)} QK Norm keys. Example: {qk_norm_keys[0]}")
        print(f"  [INFO] Shape of {qk_norm_keys[0]}: {state_dict[qk_norm_keys[0]].shape}")
    else:
        print("  [WARNING] NO QK Norm keys found in checkpoint! You should probably set qk_norm=False.")

    # 2. Check Bottleneck (MoE Gen)
    moe_keys = [k for k in state_dict.keys() if "mlp_moe_gen" in k]
    if len(moe_keys) > 0:
        print(f"  [INFO] Found {len(moe_keys)} MoE Gen keys.")
        # Check Gate Proj shape to determine intermediate size
        gate_keys = [k for k in moe_keys if "gate_proj" in k]
        if len(gate_keys) > 0:
            print(f"  [INFO] Shape of {gate_keys[0]}: {state_dict[gate_keys[0]].shape}")
            # Shape is usually [Intermediate, Hidden]
            print(f"  [INFERENCE] Detected Intermediate Size: {state_dict[gate_keys[0]].shape[0]}")
    else:
        print("  [CRITICAL] NO MoE Gen keys found in checkpoint!")

    # 3. Check Audio Projectors
    proj_keys = [k for k in state_dict.keys() if "audiovae2llm" in k]
    if len(proj_keys) > 0:
        print(f"  [INFO] Found Audio Projector keys. Shape: {state_dict[proj_keys[0]].shape}")
    else:
        print("  [CRITICAL] NO Audio Projector keys found!")
    
    print("--------------------------------------------\n")
    # --------------------------------------------------
        
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Fail fast if we didn't actually load the trained generation stack.
    # (Overfitting will never work if these are missing and silently ignored.)
    critical_prefixes = (
        "language_model.",
        "audiovae2llm.",
        "llm2audiovae.",
    )
    critical_missing = [k for k in missing if k.startswith(critical_prefixes) and "gen_pos_embed" not in k]
    if len(critical_missing) > 0:
        print(f"[CRITICAL] Missing {len(critical_missing)} critical keys (showing up to 30):")
        for k in critical_missing[:30]:
            print("  ", k)
        raise RuntimeError(
            "Checkpoint load is incomplete (critical keys missing). "
            "This usually means model config mismatched training, or you loaded the wrong checkpoint file."
        )
    
    # 验证关键权重
    moe_missing = [k for k in missing if "mlp_moe_gen" in k]
    if len(moe_missing) == 0:
        print("[SUCCESS] Bottleneck weights loaded.")
    else:
        print(f"[WARNING] Missing Bottleneck weights: {len(moe_missing)}")

    model.to(args.device, dtype=torch.bfloat16)
    model.eval()

    # 5. Load VAE
    print("Loading VAE...")
    vae_wrapper = VAEWrapper(
        model_config_path=args.vae_config,
        model_ckpt_path=args.vae_path,
        device=args.device
    )
    vae_wrapper.model.to(args.device)
    vae_wrapper.model.eval()

    # =========================
    # Trainset eval mode: dump ref / VAE recon / model sample
    # =========================
    if args.eval_dataset_yaml is not None:
        print("\n=== Trainset Eval Mode ===")
        print(f"Dataset YAML: {args.eval_dataset_yaml}")
        print(f"Num samples: {args.eval_num_samples}")
        print(f"Steps: {args.steps}, CFG: {args.cfg}")

        with open(args.eval_dataset_yaml, "r") as f:
            dataset_meta = yaml.safe_load(f)

        dataset_config = DataConfig(grouped_datasets=dataset_meta)
        packed_dataset = PackedDataset(
            dataset_config,
            tokenizer=tokenizer,
            special_tokens=new_token_ids,
            local_rank=0,
            world_size=1,
            num_workers=0,
        )
        packed_dataset.set_epoch(args.eval_seed)

        t2a_ds = None
        for ds in packed_dataset.grouped_datasets:
            if getattr(ds, "dataset_name", None) == "t2a_pretrain":
                t2a_ds = ds
                break
        if t2a_ds is None:
            raise RuntimeError(
                "Could not find a 't2a_pretrain' dataset in the provided YAML. "
                "Trainset eval mode currently expects a T2A dataset."
            )

        t2a_iter = iter(t2a_ds)
        for i in range(int(args.eval_num_samples)):
            sample = next(t2a_iter)
            audio_waveform = sample["audio_feature_list"][0]  # [C, T]
            text_ids = sample["text_ids_list"][0]
            prompt = tokenizer.decode(text_ids, skip_special_tokens=False).strip()

            out_dir = os.path.join(args.output_dir, f"trainset_{i:04d}")
            os.makedirs(out_dir, exist_ok=True)

            # Save reference audio (as processed by training audio_transform)
            _save_wav(os.path.join(out_dir, "ref.wav"), audio_waveform, int(vae_wrapper.target_sample_rate))

            # VAE recon (upper bound for overfit)
            audio_b = audio_waveform.to(args.device).unsqueeze(0)
            latents = vae_wrapper.model.encode_audio(audio_b, chunk_size=vae_wrapper.chunk_size)  # [1, D, L]
            recon = vae_wrapper.model.decode_audio(latents, chunk_size=vae_wrapper.chunk_size).squeeze(0)
            _save_wav(os.path.join(out_dir, "vae_recon.wav"), recon, int(vae_wrapper.target_sample_rate))

            # Generate with the same prompt but match latent length to the reference
            A = int(latents.shape[-1])
            soa_id = new_token_ids['start_of_audio']
            eoa_id = new_token_ids['end_of_audio']
            bos_id = new_token_ids['bos_token_id']
            eos_id = new_token_ids['eos_token_id']

            # training packs: [BOS] + text + [EOS] + [SOA] + [A audio tokens] + [EOA]
            text_ids_for_pack = [bos_id] + text_ids
            T = len(text_ids_for_pack)
            full_text_ids = text_ids_for_pack + [eos_id, soa_id, eoa_id]
            packed_text_ids = torch.tensor(full_text_ids, dtype=torch.long, device=args.device)
            text_indices = list(range(T)) + [T, T + 1, T + A + 2]
            packed_text_indexes = torch.tensor(text_indices, dtype=torch.long, device=args.device)
            packed_audio_gen_token_indexes = torch.arange(T + 2, T + 2 + A, dtype=torch.long, device=args.device)

            # Make init noise deterministic per-sample (so reruns match exactly)
            gen = torch.Generator(device=args.device)
            gen.manual_seed(int(args.seed) + i)
            packed_init_noises = torch.randn(A, model.audio_latent_dim, device=args.device, generator=gen)
            packed_audio_gen_position_ids = torch.arange(A, dtype=torch.long, device=args.device)
            total_len = T + 1 + 1 + A + 1
            packed_seqlens = torch.tensor([total_len], dtype=torch.int, device=args.device)
            pos_ids = list(range(T)) + [T] + [T + 1] * (1 + A + 1)
            packed_position_ids = torch.tensor(pos_ids, dtype=torch.long, device=args.device)
            packed_indexes = torch.arange(total_len, dtype=torch.long, device=args.device)
            key_values_lens = torch.tensor([total_len], dtype=torch.int, device=args.device)
            packed_key_value_indexes = torch.arange(total_len, dtype=torch.long, device=args.device)

            latents_list = generate_audio_standalone(
                model,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_init_noises=packed_init_noises,
                packed_audio_gen_position_ids=packed_audio_gen_position_ids,
                packed_audio_gen_token_indexes=packed_audio_gen_token_indexes,
                packed_seqlens=packed_seqlens,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                past_key_values=None,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                num_timesteps=int(args.steps),
                cfg_scale=float(args.cfg),
            )
            latent = latents_list[0].to(torch.float32).unsqueeze(0).permute(0, 2, 1)
            audio = vae_wrapper.model.decode_audio(latent, chunk_size=vae_wrapper.chunk_size).squeeze(0)
            _save_wav(os.path.join(out_dir, "model_sample.wav"), audio, int(vae_wrapper.target_sample_rate))

            with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
                f.write(prompt)

            data_idx = sample.get("data_indexes", {})
            with open(os.path.join(out_dir, "meta.txt"), "w") as f:
                f.write(str(data_idx))

            print(f"[OK] {i}: saved ref/recon/sample (A={A}) -> {out_dir}")

        print("\n[Done] Trainset eval finished.")
        return

    # 6. Define Prompts & Sweep Config
    prompts = [
        "a woman laughs and speaks while birds vocalize and water splashes", 
        "a large vehicle idling accompanied by rapid beeping noises and followed by a several artillery shots", 
        "several ducks quack while some liquid splashes", 
        "hissing noises from steam followed by a man talking",
        "a horse galloping",
        "an aircraft passes overhead as people speak",
        "a motor runs and knocks while wind blows",
        "helicopter rotors drumming and squeaking bird",
        "a loud siren whizzes past"
    ]
    
    cfg_scales = [float(args.cfg)]
    num_steps = int(args.steps)
    audio_len_latent = 215 # Approx 10 seconds

    print(f"\n=== Starting Inference Sweep ===")
    print(f"Total Prompts: {len(prompts)}")
    print(f"CFG Scales: {cfg_scales}")
    print(f"Steps: {num_steps}")
    print(f"Output Dir: {args.output_dir}")

    # 7. Run Sweep
    for p_idx, prompt in enumerate(prompts):
        print(f"\n--- Processing Prompt {p_idx+1}/{len(prompts)}: '{prompt[:50]}...' ---")
        
        # Prepare inputs (shared for all CFGs of this prompt)
        with torch.no_grad():
            text_ids = tokenizer.encode(prompt)
            soa_id = new_token_ids['start_of_audio']
            eoa_id = new_token_ids['end_of_audio']
            
            T = len(text_ids)
            A = audio_len_latent
            
            # Get Special Tokens
            bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else new_token_ids.get('bos_token_id')
            # Qwen2 usually uses <|im_end|> as EOS for chat, or standard eos_token_id
            if '<|im_end|>' in tokenizer.get_vocab():
                eos_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
            else:
                eos_id = tokenizer.eos_token_id
            
            text_ids = tokenizer.encode(prompt)
            if bos_id is not None:
                 text_ids = [bos_id] + text_ids
            
            T = len(text_ids) # Length of BOS + Text
            A = audio_len_latent
            
            # packed_text_ids: [BOS, Text...] + [EOS] + [SOA] + [EOA]
            full_text_ids = text_ids + [eos_id, soa_id, eoa_id]
            packed_text_ids = torch.tensor(full_text_ids, dtype=torch.long, device=args.device)
            
            # packed_text_indexes: Where they go in the sequence
            # Text: 0 .. T-1
            # EOS: T
            # SOA: T+1
            # EOA: T+1 + A + 1 = T + A + 2
            text_indices = list(range(T)) + [T, T + 1, T + A + 2]
            packed_text_indexes = torch.tensor(text_indices, dtype=torch.long, device=args.device)
            
            # packed_audio_gen_token_indexes: T+2 .. T+1+A
            audio_indices = list(range(T + 2, T + 2 + A))
            packed_audio_gen_token_indexes = torch.tensor(audio_indices, dtype=torch.long, device=args.device)
            
            # Fixed noise for fair comparison across CFGs
            gen = torch.Generator(device=args.device)
            gen.manual_seed(int(args.seed) + p_idx)
            packed_init_noises = torch.randn(A, model.audio_latent_dim, device=args.device, generator=gen)
            
            packed_audio_gen_position_ids = torch.arange(A, dtype=torch.long, device=args.device)
            
            # Total Length: T (Text) + 1 (EOS) + 1 (SOA) + A (Audio) + 1 (EOA)
            total_len = T + 1 + 1 + A + 1
            packed_seqlens = torch.tensor([total_len], dtype=torch.int, device=args.device)
            
            # packed_position_ids (LLM RoPE)
            # Text: 0 .. T-1
            # EOS: T
            # SOA, Audio, EOA: T+1 (All share the same position ID)
            pos_ids = list(range(T)) + [T] + [T + 1] * (1 + A + 1)
            packed_position_ids = torch.tensor(pos_ids, dtype=torch.long, device=args.device)
            
            packed_indexes = torch.arange(total_len, dtype=torch.long, device=args.device)
            key_values_lens = torch.tensor([total_len], dtype=torch.int, device=args.device)
            packed_key_value_indexes = torch.arange(total_len, dtype=torch.long, device=args.device)

            # Loop CFGs
            for cfg in cfg_scales:
                print(f"  > Generating with CFG = {cfg}")
                
                latents_list = generate_audio_standalone(
                    model,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_init_noises=packed_init_noises, # Re-use same noise
                    packed_audio_gen_position_ids=packed_audio_gen_position_ids,
                    packed_audio_gen_token_indexes=packed_audio_gen_token_indexes,
                    packed_seqlens=packed_seqlens,
                    packed_position_ids=packed_position_ids,
                    packed_indexes=packed_indexes,
                    past_key_values=None,
                    key_values_lens=key_values_lens,
                    packed_key_value_indexes=packed_key_value_indexes,
                    num_timesteps=num_steps,
                    cfg_scale=cfg,
                )
                
                # Decode and Save
                latent = latents_list[0].to(torch.float32).unsqueeze(0) # [1, T, D]
                latent = latent.permute(0, 2, 1) # VAE expects [B, D, T]
                
                # Decode with the same chunk_size + sample_rate used by training/encoding.
                audio = vae_wrapper.model.decode_audio(latent, chunk_size=vae_wrapper.chunk_size)
                audio = audio.squeeze().cpu().numpy()
                # === [FIX START] 维度转置修复 ===
                # 如果是多声道 (Shape len == 2 且 第一个维度是声道数 2)，需要转置
                # Stable Audio VAE 通常是 Stereo (2声道)
                if len(audio.shape) == 2 and audio.shape[0] < audio.shape[1]:
                    audio = audio.T  # Transpose [C, T] -> [T, C]
                # === [FIX END] =================
                safe_prompt = prompt[:20].replace(" ", "_").replace("/", "")
                filename = f"p{p_idx}_cfg{cfg}_{safe_prompt}.wav"
                save_path = os.path.join(args.output_dir, filename)
                
                sf.write(save_path, audio, int(vae_wrapper.target_sample_rate))
                print(f"    Saved: {filename}")

    print("\n[All Done] Results saved to:", args.output_dir)

if __name__ == "__main__":
    main()