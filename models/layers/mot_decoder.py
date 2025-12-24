# Copyright 2025
# Mixture-of-Transformers (MoT) Decoder Layer for Audio-Text multimodal processing
# This layer routes understanding and generation tokens through separate MLP experts

from dataclasses import dataclass
from typing import List, Optional, Tuple
import copy

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.utils import ModelOutput

from .packed_attention import PackedAttentionMoT, NaiveCache


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    """Output format for NAVIT-style packed sequence processing."""
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


class Qwen2MoTDecoderLayer(nn.Module):
    """
    Qwen2 Decoder Layer with Mixture-of-Transformers (MoT) routing.
    
    This layer implements:
    1. Dual input layer norms (one for understanding, one for generation)
    2. PackedAttentionMoT for multimodal self-attention
    3. Dual MLP experts (one for understanding audio/text, one for generation)
    4. Dual post-attention layer norms
    
    The key innovation is that understanding and generation tokens use separate
    MLP paths but share the same attention mechanism, allowing them to interact
    while maintaining specialized processing paths.
    """
    
    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        attn_module=PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.freeze_und = getattr(config, 'freeze_und', False)

        # Multi-modal self-attention (shared between understanding and generation)
        self.self_attn = attn_module(config, layer_idx)
        
        # Dual MLP experts
        self.mlp = Qwen2MLP(config)  # Understanding expert (audio, text input)
        self.mlp_moe_gen = Qwen2MLP(config)  # Generation expert (text/audio output)
        
        # Dual layer norms for input
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        gen_config = copy.deepcopy(config)
        # 尝试从配置中读取 gen_intermediate_size，如果没有则默认瘦身到 2048
        # 这对应了我们之前讨论的 "两个小矩阵相乘" 的 Bottleneck 结构
        if hasattr(config, "gen_intermediate_size"):
            gen_config.intermediate_size = config.gen_intermediate_size
        else:
            gen_config.intermediate_size = 2048  # 默认值

        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Dual layer norms for post-attention
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        print('using correct layer')
    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            output = self.forward_inference(*args, **kwargs)
            # Return as tuple for backward compatibility
            if isinstance(output, BaseNavitOutputWithPast):
                return output.packed_query_sequence, output.past_key_values
            return output

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Training forward pass with MoT routing.
        
        Args:
            packed_sequence: [total_tokens, hidden_size] - all tokens in batch
            sample_lens: List of per-sample lengths
            attention_mask: Attention mask
            packed_position_embeddings: (cos, sin) for RoPE
            packed_und_token_indexes: Indices of understanding tokens (audio, text input)
            packed_gen_token_indexes: Indices of generation tokens (text output)
        
        Returns:
            packed_sequence: Updated hidden states for all tokens
        """
        residual = packed_sequence
        
        # Apply separate input layer norms for understanding and generation tokens
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(
            packed_sequence[packed_und_token_indexes]
        )
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )

        # Self Attention (shared between understanding and generation)
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        
        # Optional: freeze understanding path gradients
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                packed_und_token_indexes
            ].detach()
        
        # Residual connection after attention
        packed_sequence = residual + packed_sequence_

        # Fully Connected (MoE routing)
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        
        # Route understanding tokens through understanding MLP expert
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                packed_und_token_indexes
            ].detach()

        # Route generation tokens through generation MLP expert
        packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        )
        
        # Residual connection after MLP
        packed_sequence = residual + packed_sequence_

        return packed_sequence

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
    ) -> Tuple[torch.Tensor, NaiveCache]:
        """
        Inference forward pass with KV caching and mode-specific routing.
        
        Args:
            mode: "und" for understanding only, "gen" for generation with mixed routing
            packed_gen_token_indexes: Indices of generation tokens (for mode="gen")
            packed_und_token_indexes: Indices of understanding tokens (for mode="gen")
        
        Returns:
            (packed_query_sequence, past_key_values): Updated hidden states and KV cache
        """
        # Handle device mismatch: if layer is on different device, move input to layer's device
        # This allows layers to stay on CPU (saving GPU memory) while inputs are on GPU
        # Store original device to move output back
        original_device = packed_query_sequence.device
        layer_device = next(self.parameters()).device
        if packed_query_sequence.device != layer_device:
            packed_query_sequence = packed_query_sequence.to(layer_device)
            # Also move position embeddings if needed
            if isinstance(packed_query_position_embeddings, tuple):
                cos, sin = packed_query_position_embeddings
                if cos.device != layer_device:
                    cos = cos.to(layer_device)
                if sin.device != layer_device:
                    sin = sin.to(layer_device)
                packed_query_position_embeddings = (cos, sin)
        
        residual = packed_query_sequence
        
        # Apply layer norm based on mode
        if mode == "und":
            # Understanding mode: all tokens use understanding layer norm
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            # Generation mode: mix of understanding (text) and generation (output) tokens
            # Ensure indexes are on the same device as packed_query_sequence
            und_idx = packed_und_token_indexes.to(packed_query_sequence.device) if packed_und_token_indexes is not None else None
            gen_idx = packed_gen_token_indexes.to(packed_query_sequence.device) if packed_gen_token_indexes is not None else None
            
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            target_dtype = packed_query_sequence.dtype
            if und_idx is not None and len(und_idx) > 0:
                packed_query_sequence_[und_idx] = self.input_layernorm(
                    packed_query_sequence[und_idx]
                ).to(dtype=target_dtype)
            if gen_idx is not None and len(gen_idx) > 0:
                packed_query_sequence_[gen_idx] = self.input_layernorm_moe_gen(
                    packed_query_sequence[gen_idx]
                ).to(dtype=target_dtype)
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_gen_token_indexes=packed_gen_token_indexes,
            packed_und_token_indexes=packed_und_token_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected (MLP routing)
        residual = packed_query_sequence
        if mode == "und":
            # Understanding mode: all tokens through understanding MLP
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            # Generation mode: route separately
            # Ensure indexes are on the same device as packed_query_sequence
            und_idx = packed_und_token_indexes.to(packed_query_sequence.device) if packed_und_token_indexes is not None else None
            gen_idx = packed_gen_token_indexes.to(packed_query_sequence.device) if packed_gen_token_indexes is not None else None
            
            packed_und_query_sequence = packed_query_sequence[und_idx] if und_idx is not None and len(und_idx) > 0 else None
            packed_gen_query_sequence = packed_query_sequence[gen_idx] if gen_idx is not None and len(gen_idx) > 0 else None
            
            target_dtype = packed_query_sequence.dtype
            
            if packed_und_query_sequence is not None:
                packed_und_query_sequence = self.post_attention_layernorm(
                    packed_und_query_sequence
                ).to(dtype=target_dtype)
            if packed_gen_query_sequence is not None:
                packed_gen_query_sequence = self.post_attention_layernorm_moe_gen(
                    packed_gen_query_sequence
                ).to(dtype=target_dtype)

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            if und_idx is not None and len(und_idx) > 0 and packed_und_query_sequence is not None:
                # Ensure input dtype matches MLP weight dtype
                mlp_dtype = next(self.mlp.parameters()).dtype
                und_mlp_input = packed_und_query_sequence.to(dtype=mlp_dtype) if packed_und_query_sequence.dtype != mlp_dtype else packed_und_query_sequence
                packed_query_sequence_[und_idx] = self.mlp(und_mlp_input).to(dtype=target_dtype)
            if gen_idx is not None and len(gen_idx) > 0 and packed_gen_query_sequence is not None:
                # Ensure input dtype matches MLP weight dtype
                mlp_gen_dtype = next(self.mlp_moe_gen.parameters()).dtype
                gen_mlp_input = packed_gen_query_sequence.to(dtype=mlp_gen_dtype) if packed_gen_query_sequence.dtype != mlp_gen_dtype else packed_gen_query_sequence
                
                gen_output = self.mlp_moe_gen(gen_mlp_input)
                
                # --- Debug Bottleneck ---
                if self.layer_idx is not None and self.layer_idx in [0, 31]:
                    print(f"[Debug Layer {self.layer_idx}] Gen MLP Intermediate Size: {self.mlp_moe_gen.gate_proj.out_features}")
                    # Check if outputs are identical across assumed batch split
                    total_gen_tokens = gen_output.shape[0]
                    if total_gen_tokens % 4 == 0: # Assuming batch size 4
                        split_size = total_gen_tokens // 4
                        s0 = gen_output[0:split_size].float().mean()
                        s1 = gen_output[split_size:2*split_size].float().mean()
                        print(f"[Debug Layer {self.layer_idx}] Gen Output Means - S0: {s0:.6f}, S1: {s1:.6f} (Diff: {abs(s0-s1):.6f})")
                # ------------------------

                packed_query_sequence_[gen_idx] = gen_output.to(dtype=target_dtype)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence
        
        # Move output back to original device if layer was on different device
        if packed_query_sequence.device != original_device:
            packed_query_sequence = packed_query_sequence.to(original_device)

        # Return as BaseNavitOutputWithPast object for consistency
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values
        )

