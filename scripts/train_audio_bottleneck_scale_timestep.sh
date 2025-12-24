#!/bin/bash

# Set up environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_API_KEY="be9d7aae372d439f9b8cb98aeab5d9ba96dcf7f3"
export WANDB_DISABLED=True
export WANDB_MODE=offline


# Path to the Qwen2-Audio-7B-Instruct model
# Assuming it is located at ../Models/Qwen2-Audio-7B-Instruct relative to the project root
MODEL_PATH="/mnt/cfs/5vr0p6/yimingjing/workspace/Models/Qwen2-Audio-7B-Instruct"

# Output directory
OUTPUT_DIR="results/audio_bottleneck_training_overfit_scale_timestep"

# Run training
torchrun --nproc_per_node=4 train/pretrain_unified_navit.py \
    --model_path "hf/BAGEL-7B-MoT" \
    --llm_path "$MODEL_PATH" \
    --qwen2_audio_path "$MODEL_PATH" \
    --audio_gen True \
    --audio_und True \
    --gen_intermediate_size 2048 \
    --online_audio_encoding True \
    --audio_vae_path "ckpt/stable_audio_open_vae_weights.pth" \
    --audio_vae_config "configs/audio_vae.json" \
    --results_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
    --num_shard 4 \
    --dataset_config_file "data/configs/example.yaml" \
    --num_workers 4 \
    --total_steps 50000 \
    --save_every 1000 \
    --log_every 10 \
    --lr 1e-3 \
    --gradient_accumulation_steps 1 \
    --max_num_tokens_per_sample 2048 \
    --max_num_tokens 8192 \
    --expected_num_tokens 8192 \
