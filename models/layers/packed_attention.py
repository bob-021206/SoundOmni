# Copyright 2025
# Packed Attention Layer adapted from BAGEL for Audio-Text processing
# This allows understanding (audio) and generation (text/future audio) tokens
# to be processed in the same sequence with different expert paths.

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
)

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, will use slower attention")

# Flag to track if Flash Attention warning has been printed
_FLASH_ATTN_WARNING_PRINTED = False


class NaiveCache:
    """Simple KV cache for inference."""
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


class PackedAttentionMoT(Qwen2Attention):
    """
    Packed Attention with Mixture-of-Transformers (MoT) for audio-text multimodal processing.
    
    This layer supports:
    - Understanding tokens (audio + text input): use standard Q/K/V projections
    - Generation tokens (text/audio output): use separate moe_gen Q/K/V projections
    - Both token types can attend to each other in the same sequence
    
    Key differences from standard attention:
    1. Dual Q/K/V/O projections (one for understanding, one for generation)
    2. Dual QK normalization layers
    3. Packed sequence processing with variable-length samples
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # Store config for easy access
        self.config = config
        
        # Get dimensions from config and explicitly set as instance attributes
        # (in case parent class doesn't set them as attributes)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = hidden_size // num_heads
        
        # Explicitly store as instance attributes for use in forward methods
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_key_value_heads
        
        # QK normalization for understanding path
        if getattr(config, 'qk_norm', False):
            self.q_norm = Qwen2RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(head_dim, eps=config.rms_norm_eps)
            # QK normalization for generation path
            self.q_norm_moe_gen = Qwen2RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        # Generation expert projections
        self.q_proj_moe_gen = nn.Linear(
            hidden_size, num_heads * head_dim, bias=True
        )
        self.k_proj_moe_gen = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=True
        )
        self.v_proj_moe_gen = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=True
        )
        self.o_proj_moe_gen = nn.Linear(
            num_heads * head_dim, hidden_size, bias=False
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        """
        Training forward pass with packed sequences.
        
        Args:
            packed_sequence: [total_tokens, hidden_size] - concatenated tokens from all samples
            sample_lens: List of lengths for each sample in the batch
            attention_mask: Attention mask (can be List or BlockMask)
            packed_position_embeddings: (cos, sin) for RoPE
            packed_und_token_indexes: Indices of understanding tokens (audio, text input)
            packed_gen_token_indexes: Indices of generation tokens (text output)
        """
        # Initialize Q/K/V for all tokens
        packed_query_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_heads * self.head_dim)
        )
        packed_key_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )
        packed_value_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )

        # Split sequence into understanding and generation parts
        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        # Apply understanding expert projections
        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)

        # Apply generation expert projections
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)

        # Reshape for multi-head attention
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        # Optional: freeze understanding path during training
        if getattr(self.config, 'freeze_und', False):
            packed_value_states[packed_und_token_indexes] = packed_value_states[
                packed_und_token_indexes
            ].detach()

        # Apply QK normalization separately for understanding and generation
        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(
            packed_query_states[packed_und_token_indexes]
        )
        if getattr(self.config, 'freeze_und', False):
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[
                packed_und_token_indexes
            ].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_gen_token_indexes]
        )

        packed_key_states_[packed_und_token_indexes] = self.k_norm(
            packed_key_states[packed_und_token_indexes]
        )
        if getattr(self.config, 'freeze_und', False):
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[
                packed_und_token_indexes
            ].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(
            packed_key_states[packed_gen_token_indexes]
        )

        # Apply RoPE
        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        # Perform attention (using efficient attention or flash attention)
        if isinstance(attention_mask, List):
            # Use efficient attention with per-sample masks
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            # Use block mask (more efficient for longer sequences)
            raise NotImplementedError("Block mask attention not yet implemented")

        # Reshape and apply output projection
        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
            -1, self.num_heads * self.head_dim
        )
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        
        # Route through understanding or generation output projection
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(
            packed_attn_output[packed_und_token_indexes]
        )
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(
            packed_attn_output[packed_gen_token_indexes]
        )

        return packed_attn_output_

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_gen_token_indexes=None,
        packed_und_token_indexes=None,
    ):
        """
        Inference forward pass with KV caching.
        
        Args:
            mode: "und" for understanding, "gen" for generation
            packed_gen_token_indexes: Indices of generation tokens (for mode="gen")
            packed_und_token_indexes: Indices of understanding tokens (for mode="gen")
        """
        # Prepare indexes for generation mode (ensure they're on correct device)
        und_idx = None
        gen_idx = None
        target_dtype = packed_query_sequence.dtype
        if mode == 'gen':
            und_idx = packed_und_token_indexes.to(packed_query_sequence.device) if packed_und_token_indexes is not None else None
            gen_idx = packed_gen_token_indexes.to(packed_query_sequence.device) if packed_gen_token_indexes is not None else None
        
        if mode == 'und':
            # Understanding mode: use standard projections
            packed_query_states = self.q_proj(packed_query_sequence).view(
                -1, self.num_heads, self.head_dim
            )
            packed_key_states = self.k_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_value_states = self.v_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
            
        elif mode == 'gen':
            # Generation mode: use mixed projections (text uses und, gen tokens use moe_gen)
            packed_query_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
            )
            packed_key_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            packed_value_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )

            packed_und_query_sequence = packed_query_sequence[und_idx] if und_idx is not None and len(und_idx) > 0 else None
            packed_gen_query_sequence = packed_query_sequence[gen_idx] if gen_idx is not None and len(gen_idx) > 0 else None

            if packed_und_query_sequence is not None:
                # Ensure input dtype matches projection layer weight dtype
                q_proj_dtype = next(self.q_proj.parameters()).dtype
                und_input = packed_und_query_sequence.to(dtype=q_proj_dtype) if packed_und_query_sequence.dtype != q_proj_dtype else packed_und_query_sequence
                packed_query_states[und_idx] = self.q_proj(und_input).to(dtype=target_dtype)
            if packed_gen_query_sequence is not None:
                # Ensure input dtype matches projection layer weight dtype
                q_proj_gen_dtype = next(self.q_proj_moe_gen.parameters()).dtype
                gen_input = packed_gen_query_sequence.to(dtype=q_proj_gen_dtype) if packed_gen_query_sequence.dtype != q_proj_gen_dtype else packed_gen_query_sequence
                packed_query_states[gen_idx] = self.q_proj_moe_gen(gen_input).to(dtype=target_dtype)

            if packed_und_query_sequence is not None:
                # Ensure input dtype matches projection layer weight dtype
                k_proj_dtype = next(self.k_proj.parameters()).dtype
                und_input = packed_und_query_sequence.to(dtype=k_proj_dtype) if packed_und_query_sequence.dtype != k_proj_dtype else packed_und_query_sequence
                packed_key_states[und_idx] = self.k_proj(und_input).to(dtype=target_dtype)
            if packed_gen_query_sequence is not None:
                # Ensure input dtype matches projection layer weight dtype
                k_proj_gen_dtype = next(self.k_proj_moe_gen.parameters()).dtype
                gen_input = packed_gen_query_sequence.to(dtype=k_proj_gen_dtype) if packed_gen_query_sequence.dtype != k_proj_gen_dtype else packed_gen_query_sequence
                packed_key_states[gen_idx] = self.k_proj_moe_gen(gen_input).to(dtype=target_dtype)

            if packed_und_query_sequence is not None:
                # Ensure input dtype matches projection layer weight dtype
                v_proj_dtype = next(self.v_proj.parameters()).dtype
                und_input = packed_und_query_sequence.to(dtype=v_proj_dtype) if packed_und_query_sequence.dtype != v_proj_dtype else packed_und_query_sequence
                packed_value_states[und_idx] = self.v_proj(und_input).to(dtype=target_dtype)
            if packed_gen_query_sequence is not None:
                # Ensure input dtype matches projection layer weight dtype
                v_proj_gen_dtype = next(self.v_proj_moe_gen.parameters()).dtype
                gen_input = packed_gen_query_sequence.to(dtype=v_proj_gen_dtype) if packed_gen_query_sequence.dtype != v_proj_gen_dtype else packed_gen_query_sequence
                packed_value_states[gen_idx] = self.v_proj_moe_gen(gen_input).to(dtype=target_dtype)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(
                -1, self.num_key_value_heads, self.head_dim
            )

            # Apply normalization (layer norm may output float32, convert to target_dtype)
            if und_idx is not None and len(und_idx) > 0:
                packed_query_states[und_idx] = self.q_norm(
                    packed_query_states[und_idx]
                ).to(dtype=target_dtype)
            if gen_idx is not None and len(gen_idx) > 0:
                packed_query_states[gen_idx] = self.q_norm_moe_gen(
                    packed_query_states[gen_idx]
                ).to(dtype=target_dtype)

            if und_idx is not None and len(und_idx) > 0:
                packed_key_states[und_idx] = self.k_norm(
                    packed_key_states[und_idx]
                ).to(dtype=target_dtype)
            if gen_idx is not None and len(gen_idx) > 0:
                packed_key_states[gen_idx] = self.k_norm_moe_gen(
                    packed_key_states[gen_idx]
                ).to(dtype=target_dtype)

        # Apply RoPE
        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        # Merge with cached KV if available
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(
                size=[seqlens, self.num_key_value_heads, self.head_dim]
            )
            merged_value_states = past_key_states.new_zeros(
                size=[seqlens, self.num_key_value_heads, self.head_dim]
            )
            
            # Debug: Check indexes for first layer and first generation step
            if hasattr(self, '_debug_generation') and self._debug_generation and self.layer_idx == 0:
                print(f"      [packed_attention] KV cache merge:")
                print(f"        seqlens={seqlens}, query_lens={query_lens.tolist()}, key_values_lens={key_values_lens.tolist()}")
                print(f"        past_key_states shape: {past_key_states.shape}")
                print(f"        packed_key_states shape: {packed_key_states.shape}")
                print(f"        packed_query_indexes: {packed_query_indexes.tolist() if len(packed_query_indexes) <= 10 else 'too long'}")
                print(f"        packed_key_value_indexes: {packed_key_value_indexes[:10].tolist() if len(packed_key_value_indexes) > 10 else packed_key_value_indexes.tolist()}... (len={len(packed_key_value_indexes)})")
                print(f"        Expected: packed_key_value_indexes should be [0, 1, 2, ..., {key_values_lens[0].item()-1}]")
                if len(packed_key_value_indexes) > 0:
                    print(f"        Actual first 5: {packed_key_value_indexes[:5].tolist()}")
                    print(f"        Actual last 5: {packed_key_value_indexes[-5:].tolist()}")
            
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        # Use flash attention for inference if available, otherwise fallback to standard attention
        try:
            if FLASH_ATTN_AVAILABLE:
                cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
                cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

                packed_attn_output = flash_attn_varlen_func(
                    q=packed_query_states,
                    k=merged_key_states,
                    v=merged_value_states,
                    cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                    cu_seqlens_k=cu_seqlens_k.to(torch.int32),
                    max_seqlen_q=max(query_lens).item(),
                    max_seqlen_k=max(key_values_lens).item(),
                    causal=is_causal,
                )
            else:
                raise RuntimeError("Flash attention not available, using fallback")
        except (RuntimeError, Exception) as e:
            # Fallback to standard PyTorch attention
            # Only print warning once per session
            global _FLASH_ATTN_WARNING_PRINTED
            if not _FLASH_ATTN_WARNING_PRINTED:
                print(f"⚠️  Flash Attention failed ({str(e)}), using standard attention fallback")
                _FLASH_ATTN_WARNING_PRINTED = True
            
            # Expand KV for GQA if needed
            if self.num_key_value_groups > 1:
                merged_key_states = merged_key_states[:, :, None, :].repeat(
                    1, 1, self.num_key_value_groups, 1
                ).reshape(-1, self.num_heads, self.head_dim)
                merged_value_states = merged_value_states[:, :, None, :].repeat(
                    1, 1, self.num_key_value_groups, 1
                ).reshape(-1, self.num_heads, self.head_dim)
            
            # Split into per-sample sequences
            query_splits = torch.split(packed_query_states, query_lens.tolist(), dim=0)
            key_splits = torch.split(merged_key_states, key_values_lens.tolist(), dim=0)
            value_splits = torch.split(merged_value_states, key_values_lens.tolist(), dim=0)
            
            attn_outputs = []
            for q, k, v in zip(query_splits, key_splits, value_splits):
                # q: [seq_len_q, num_heads, head_dim]
                # k, v: [seq_len_kv, num_heads, head_dim]
                
                # Reshape for scaled_dot_product_attention: [batch=1, num_heads, seq_len, head_dim]
                q = q.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len_q, head_dim]
                k = k.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len_kv, head_dim]
                v = v.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len_kv, head_dim]
                
                # Compute attention
                attn_out = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                
                # Reshape back: [1, num_heads, seq_len_q, head_dim] -> [seq_len_q, num_heads, head_dim]
                attn_out = attn_out.squeeze(0).transpose(0, 1)
                attn_outputs.append(attn_out)
            
            # Concatenate all samples
            packed_attn_output = torch.cat(attn_outputs, dim=0)

        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        
        if mode == 'und':
            # Understanding mode: ensure dtype matches projection layer weights
            o_proj_dtype = next(self.o_proj.parameters()).dtype
            if packed_attn_output.dtype != o_proj_dtype:
                packed_attn_output = packed_attn_output.to(dtype=o_proj_dtype)
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'gen':
            # Generation mode: create output tensor with target_dtype first
            # This ensures index assignment works correctly
            packed_attn_output_final = torch.zeros_like(packed_attn_output).to(dtype=target_dtype)
            
            if und_idx is not None and len(und_idx) > 0:
                # Ensure input dtype matches projection layer weight dtype
                o_proj_dtype = next(self.o_proj.parameters()).dtype
                und_attn = packed_attn_output[und_idx]
                if und_attn.dtype != o_proj_dtype:
                    und_attn = und_attn.to(dtype=o_proj_dtype)
                packed_attn_output_final[und_idx] = self.o_proj(und_attn).to(dtype=target_dtype)
            if gen_idx is not None and len(gen_idx) > 0:
                # Ensure input dtype matches projection layer weight dtype
                o_proj_gen_dtype = next(self.o_proj_moe_gen.parameters()).dtype
                gen_attn = packed_attn_output[gen_idx]
                if gen_attn.dtype != o_proj_gen_dtype:
                    gen_attn = gen_attn.to(dtype=o_proj_gen_dtype)
                packed_attn_output_final[gen_idx] = self.o_proj_moe_gen(gen_attn).to(dtype=target_dtype)
            
            packed_attn_output = packed_attn_output_final

        if update_past_key_values and past_key_values is not None:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values

