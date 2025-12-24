from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
import numpy as np
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

from .layers.mot_decoder import Qwen2MoTDecoderLayer, BaseNavitOutputWithPast
from .layers.packed_attention import NaiveCache


class Qwen2AudioBagelConfig(PretrainedConfig):
    """
    Configuration class for Qwen2AudioBagel model.
    
    This config combines:
    - Audio encoder settings from Qwen2-Audio
    - LLM settings from Qwen2
    - MoT-specific settings from BAGEL
    """
    model_type = "qwen2_audio_bagel"
    
    def __init__(
        self,
        # Audio encoder config
        audio_config=None,
        audio_encoder_name="openai/whisper-large-v3",
        
        # LLM config (Qwen2-style)
        vocab_size=151936,
        hidden_size=3584,  # Qwen2-Audio-7B uses 3584
        intermediate_size=18944,
        gen_intermediate_size=2048, # Bottleneck size for generation experts
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        
        # MoT-specific config
        qk_norm=True,
        freeze_und=False,
        use_moe=True,  # Enable MoT routing
        
        # Audio-text integration
        audio_to_text_projection_hidden_size=None,  # If None, uses hidden_size
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Audio settings
        self.audio_config = audio_config
        self.audio_encoder_name = audio_encoder_name
        
        # LLM settings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gen_intermediate_size = gen_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        
        # MoT settings
        self.qk_norm = qk_norm
        self.freeze_und = freeze_und
        self.use_moe = use_moe
        
        # Projection settings
        self.audio_to_text_projection_hidden_size = (
            audio_to_text_projection_hidden_size or hidden_size
        )


class Qwen2Model(Qwen2PreTrainedModel):
    """
    Qwen2 Model with MoT Decoder Layers.
    
    This is the core LLM that processes the unified audio-text sequence.
    It uses Qwen2MoTDecoderLayer to route understanding and generation tokens
    through separate expert paths.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = getattr(config, 'use_moe', True)

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # MoT decoder layers
        self.layers = nn.ModuleList(
            [Qwen2MoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Dual output layer norms (for understanding and generation)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # RoPE
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights
        self.post_init()

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
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Training forward with packed sequences and MoT routing."""
        
        if getattr(self.config, 'freeze_und', False):
            packed_sequence[packed_und_token_indexes] = packed_sequence[
                packed_und_token_indexes
            ].detach()

        # Create position embeddings for RoPE
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        # Prepare MoE inputs
        extra_inputs = {}
        if self.use_moe:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        # Pass through all decoder layers
        for decoder_layer in self.layers:
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

        # Apply final layer norm (separate for understanding and generation)
        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(
                packed_sequence[packed_und_token_indexes]
            )
            if getattr(self.config, 'freeze_und', False):
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                    packed_und_token_indexes
                ].detach()
            packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(
                packed_sequence[packed_gen_token_indexes]
            )
            return packed_sequence_
        else:
            return self.norm(packed_sequence)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_gen_token_indexes=None,
        packed_und_token_indexes=None,
    ) -> BaseNavitOutputWithPast:
        """Inference forward with KV caching and mode-specific routing."""
        
        # Create position embeddings for RoPE
        # Debug: Check position IDs before rotary_emb
        if hasattr(self, '_debug_generation') and self._debug_generation:
            print(f"      [forward_inference] packed_query_position_ids: {packed_query_position_ids.tolist() if packed_query_position_ids.numel() <= 10 else 'too long'}, shape: {packed_query_position_ids.shape}")
            print(f"      [forward_inference] packed_query_sequence shape: {packed_query_sequence.shape}")
        
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)
        
        # Debug: Check rotary embeddings
        if hasattr(self, '_debug_generation') and self._debug_generation:
            print(f"      [forward_inference] cos shape: {cos.shape}, sin shape: {sin.shape}")

        # Prepare MoE inputs based on mode
        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert packed_gen_token_indexes is not None
                assert packed_und_token_indexes is not None
                extra_inputs.update(
                    packed_gen_token_indexes=packed_gen_token_indexes,
                    packed_und_token_indexes=packed_und_token_indexes,
                )

        # Pass through all decoder layers
        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        # Apply final layer norm
        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                # Ensure inputs are on the same device as the norm layers
                norm_device = next(self.norm.parameters()).device
                norm_moe_gen_device = next(self.norm_moe_gen.parameters()).device
                target_device = packed_query_sequence.device
                target_dtype = packed_query_sequence.dtype
                
                # Process understanding tokens
                if len(packed_und_token_indexes) > 0:
                    und_tokens = packed_query_sequence[packed_und_token_indexes]
                    if und_tokens.device != norm_device:
                        und_tokens = und_tokens.to(device=norm_device)
                    und_output = self.norm(und_tokens)
                    # Move output back to target device and dtype
                    if und_output.device != target_device:
                        und_output = und_output.to(device=target_device)
                    if und_output.dtype != target_dtype:
                        und_output = und_output.to(dtype=target_dtype)
                    packed_query_sequence_[packed_und_token_indexes] = und_output
                
                # Process generation tokens
                if len(packed_gen_token_indexes) > 0:
                    gen_tokens = packed_query_sequence[packed_gen_token_indexes]
                    if gen_tokens.device != norm_moe_gen_device:
                        gen_tokens = gen_tokens.to(device=norm_moe_gen_device)
                    gen_output = self.norm_moe_gen(gen_tokens)
                    # Move output back to target device and dtype
                    if gen_output.device != target_device:
                        gen_output = gen_output.to(device=target_device)
                    if gen_output.dtype != target_dtype:
                        gen_output = gen_output.to(dtype=target_dtype)
                    packed_query_sequence_[packed_gen_token_indexes] = gen_output
                
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    """Qwen2 Causal LM with LM head for text generation."""
    
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def init_moe(self):
        """Initialize MoE generation experts by copying from understanding experts."""
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_gen_token_indexes=None,
        packed_und_token_indexes=None,
    ) -> BaseNavitOutputWithPast:
        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
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
        return outputs