# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from torch import distributed as dist

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)

from .layers.packed_attention import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding, get_1d_sincos_pos_embed_from_grid
from .cache_utils.taylorseer import cache_init
from modeling.qwen2.modeling_qwen2 import Qwen2MLP

from tqdm import tqdm


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        audio_und=True,  # Audio understanding support
        audio_gen=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        audio_config=None,  # Audio encoder config

        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        # timestep_shift=1.0,
        audio_encoder_hidden_size=1280,  # Whisper Large-v3 output dim

        # --- 生成分支参数---
        audio_latent_dim=64,       # VAE 维度
        max_source_audio_len=3000, # 理解侧最大长度
        max_target_audio_len=2048, # 生成侧最大长度 (用于 1D Pos Embed)
        timestep_shift=1.0,        # Flow Matching 参数

        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.audio_und = audio_und  # Audio understanding flag
        self.audio_gen = audio_gen
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.audio_config = audio_config  # Audio config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.audio_encoder_hidden_size = audio_encoder_hidden_size
        self.audio_latent_dim = audio_latent_dim
        self.max_source_audio_len = max_source_audio_len
        self.max_target_audio_len = max_target_audio_len


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        # Audio understanding components (from Qwen2AudioBagel)
        if config.audio_und:
            self.audio_encoder = None  # To be loaded from Qwen2-Audio
            self.audio_encoder_hidden_size = config.audio_encoder_hidden_size
            self.audio_projector = nn.Linear(
                self.audio_encoder_hidden_size,
                self.hidden_size,
                bias=True
            )
            # Reference model for generation fallback
            self._audio_qwen2_ref_model = None
        
        if config.audio_gen:
            self.audio_latent_dim = config.audio_latent_dim
            self.timestep_shift = config.timestep_shift
            
            # 投影层：连接 VAE 空间与 LLM 空间
            self.audiovae2llm = nn.Linear(self.audio_latent_dim, config.llm_config.hidden_size)
            self.llm2audiovae = nn.Linear(config.llm_config.hidden_size, self.audio_latent_dim)
            
            # 时间步嵌入
            self.time_embedder = TimestepEmbedder(config.llm_config.hidden_size)
            
            # [重点] 1D 位置编码
            # 使用 1D sin/cos position embedding
            pos_embed = get_1d_sincos_pos_embed_from_grid(config.llm_config.hidden_size, np.arange(config.max_target_audio_len))
            self.register_buffer("gen_pos_embed", torch.from_numpy(pos_embed).float())
            
            # 初始化权重 (仅生成部分)
            # self._init_gen_weights()
        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)
        if self.config.audio_gen:
            nn.init.normal_(self.audiovae2llm.weight, std=0.02)
            nn.init.constant_(self.llm2audiovae.weight, 0)
            nn.init.constant_(self.llm2audiovae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        # for audio generation
        packed_audio_latents: Optional[torch.Tensor] = None,
        packed_audio_gen_token_indexes: Optional[torch.LongTensor] = None,
        packed_audio_gen_position_ids: Optional[torch.LongTensor] = None,
        packed_audio_timesteps: Optional[torch.LongTensor] = None,
        audio_mse_loss_indexes: Optional[torch.BoolTensor] = None,
        # for audio understanding
        packed_audio_features: Optional[torch.Tensor] = None,
        packed_audio_und_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.

            packed_audio_latents: 1-D tensor, flattened audio latents [Total_Audio_Tokens, D].
            packed_audio_gen_token_indexes: 1-D int tensor, indexes in packed_sequence for audio gen tokens.
            packed_audio_gen_position_ids: 1-D int tensor, position ids for audio gen tokens.
            packed_audio_timesteps: 1-D float tensor, flow timesteps for audio.
            audio_mse_loss_indexes: 1-D bool tensor, where to compute audio mse loss.
            
            packed_audio_features: Audio features for understanding.
            packed_audio_und_token_indexes: Indexes for audio understanding tokens.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        # Vis Understanding branch
        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        # Audio Understanding branch
        if self.config.audio_und and packed_audio_features is not None:
            packed_audio_embeds = self.audio_encode(packed_audio_features)
            if packed_audio_embeds.dim() == 3:
                packed_audio_embeds = packed_audio_embeds.reshape(-1, self.hidden_size)
            packed_sequence[packed_audio_und_token_indexes] = packed_audio_embeds

        if self.config.audio_gen and packed_audio_latents is not None:
            # Audio generation branch
            noise = torch.randn_like(packed_audio_latents)
            # # === [DEBUG 代码 START] ===
            # # 打印进入 Sigmoid 之前的数据分布
            # if dist.get_rank() == 0: # 只让主进程打印，防止刷屏
            #     t_min = packed_audio_timesteps.min().item()
            #     t_max = packed_audio_timesteps.max().item()
            #     print(f"\n[DEBUG] Raw Timesteps -> Min: {t_min:.4f}, Max: {t_max:.4f}")
            # # === [DEBUG 代码 END] ===

            # # # add scale factor to ensure the input range covers more of the sigmoid curve
            # # scale_factor = 5.0 
            # # packed_audio_timesteps = packed_audio_timesteps * scale_factor

            # Match visual branch behavior: timesteps are sampled in dataset as N(0,1)
            # and squashed to (0,1) here. Also handles -inf -> 0 cleanly.
            packed_audio_timesteps = torch.sigmoid(packed_audio_timesteps)
            # # === [DEBUG 代码 START] ===
            # # 打印 Sigmoid 之后的数据分布
            # if dist.get_rank() == 0:
            #     t_post_min = packed_audio_timesteps.min().item()
            #     t_post_max = packed_audio_timesteps.max().item()
            #     print(f"[DEBUG] After Sigmoid -> Min: {t_post_min:.4f}, Max: {t_post_max:.4f}\n")
            # # === [DEBUG 代码 END] ===
            packed_audio_timesteps = self.timestep_shift * packed_audio_timesteps / (1 + (self.timestep_shift - 1) * packed_audio_timesteps)
            
            x_t = (1 - packed_audio_timesteps[:, None]) * packed_audio_latents + packed_audio_timesteps[:, None] * noise
            
            hidden_x_t = self.audiovae2llm(x_t)
            time_emb = self.time_embedder(packed_audio_timesteps)
            pos_emb = self.gen_pos_embed[packed_audio_gen_position_ids]
            
            audio_gen_input = hidden_x_t + time_emb + pos_emb
            
            packed_sequence[packed_audio_gen_token_indexes] = audio_gen_input
            
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_und_token_indexes, packed_vit_token_indexes], dim=0)
            if packed_audio_und_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_und_token_indexes, packed_audio_und_token_indexes], dim=0)
            
            packed_gen_token_indexes = packed_vae_token_indexes
            if packed_audio_gen_token_indexes is not None:
                if packed_gen_token_indexes is not None:
                    packed_gen_token_indexes = torch.cat([packed_gen_token_indexes, packed_audio_gen_token_indexes], dim=0)
                else:
                    packed_gen_token_indexes = packed_audio_gen_token_indexes

            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )        
            last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen and mse_loss_indexes is not None:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2
            
        audio_mse = None
        if self.config.audio_gen and audio_mse_loss_indexes is not None:
            packed_audio_preds = self.llm2audiovae(last_hidden_state[audio_mse_loss_indexes])
            audio_target = noise - packed_audio_latents
            has_audio_mse = packed_audio_timesteps > 0
            
            audio_mse = (packed_audio_preds[has_audio_mse] - audio_target[has_audio_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, audio_mse=audio_mse, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            # text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            # Don't add EOS, let the user/chat template handle it. 
            # Also BOS might be redundant if prepare_start_tokens adds it, but let's keep BOS for now as it marks start of text segment?
            # Actually, if we have multiple segments, we don't want BOS everywhere.
            # But for now, let's just remove EOS.
            # text_ids = [new_token_ids['bos_token_id']] + text_ids
            # REMOVED BOS as well to avoid confusing Qwen2-Audio which expects <|im_start|> directly
            pass
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        if past_key_values is None:
            past_key_values = NaiveCache(self.language_model.config.num_hidden_layers)

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_audio_latents(self, curr_kvlens, curr_rope, audio_lengths, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_audio_gen_position_ids, packed_audio_gen_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for audio_len, curr_kvlen, curr_position_id in zip(audio_lengths, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # Start of Audio Token
            packed_text_ids.append(new_token_ids.get('start_of_audio', new_token_ids.get('bos_token_id'))) # Fallback if not defined
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            # Audio Position IDs (1D)
            # Assuming audio_len is the number of latent tokens
            audio_pos_ids = torch.arange(audio_len, dtype=torch.long)
            packed_audio_gen_position_ids.append(audio_pos_ids)

            # Init Noise
            packed_init_noises.append(
                torch.randn(audio_len, self.audio_latent_dim)
            )
            
            packed_audio_gen_token_indexes.extend(range(query_curr, query_curr + audio_len))
            packed_indexes.extend(range(curr, curr + audio_len))
            curr += audio_len
            query_curr += audio_len

            # End of Audio Token
            packed_text_ids.append(new_token_ids.get('end_of_audio', new_token_ids.get('eos_token_id')))
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (audio_len + 2))
            packed_seqlens.append(audio_len + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_audio_gen_position_ids": torch.cat(packed_audio_gen_position_ids, dim=0),
            "packed_audio_gen_token_indexes": torch.tensor(packed_audio_gen_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
    ):
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
    
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
        
        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        
        if self.language_model.model.enable_taylorseer:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    @torch.no_grad
    def generate_audio(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_audio_gen_position_ids: torch.LongTensor,
        packed_audio_gen_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
    ):
        x_t = packed_init_noises.to(dtype=self.audiovae2llm.weight.dtype)

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=dist.is_initialized() and dist.get_rank() != 0):
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] generate_audio step {i} start", flush=True)
            timestep = torch.full((1,), t, device=x_t.device)
            
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
            else:
                cfg_text_scale_ = 1.0

            print(f"[Rank {rank}] calling _forward_audio_flow", flush=True)
            v_t = self._forward_audio_flow(
                x_t=x_t,
                timestep=timestep,
                packed_audio_gen_token_indexes=packed_audio_gen_token_indexes,
                packed_audio_gen_position_ids=packed_audio_gen_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_indexes=packed_indexes,
                packed_position_ids=packed_position_ids,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
            )
            print(f"[Rank {rank}] _forward_audio_flow done", flush=True)
            
            x_t = x_t - v_t * dts[i]

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_audio_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_audio_gen_token_indexes: torch.LongTensor,
        packed_audio_gen_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Inside _forward_audio_flow", flush=True)
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # Audio Embeddings
        hidden_x_t = self.audiovae2llm(x_t)
        time_emb = self.time_embedder(timestep)
        pos_emb = self.gen_pos_embed[packed_audio_gen_position_ids]
        
        audio_gen_input = hidden_x_t + time_emb + pos_emb
        if audio_gen_input.dtype != packed_sequence.dtype:
            audio_gen_input = audio_gen_input.to(packed_sequence.dtype)
        packed_sequence[packed_audio_gen_token_indexes] = audio_gen_input

        extra_inputs = {}
        if self.use_moe:
             extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_audio_gen_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        print(f"[Rank {rank}] Calling language_model.forward_inference", flush=True)
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        print(f"[Rank {rank}] language_model.forward_inference returned", flush=True)
        
        audio_output = output.packed_query_sequence[packed_audio_gen_token_indexes]

        # --- DEBUG: Check intermediate features ---
        if rank == 0:
            B = len(key_values_lens)
            total_tokens = audio_output.shape[0]
            if total_tokens % B == 0:
                tokens_per_sample = total_tokens // B
                print(f"[Debug] Audio Output Features (Step {timestep[0].item():.2f}):")
                for b in range(B):
                    sample_feat = audio_output[b*tokens_per_sample : (b+1)*tokens_per_sample]
                    print(f"  Sample {b} Feat Mean: {sample_feat.mean().item():.6f}, Std: {sample_feat.std().item():.6f}")
        # ------------------------------------------

        v_t = self.llm2audiovae(audio_output)

        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids if cfg_text_packed_position_ids is not None else packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes if cfg_text_packed_query_indexes is not None else packed_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_audio_output = cfg_text_output.packed_query_sequence[packed_audio_gen_token_indexes]
            cfg_text_v_t = self.llm2audiovae(cfg_text_audio_output)

            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

            if cfg_renorm_type == "global":
                norm_v_t = torch.norm(v_t)
                norm_v_t_text_ = torch.norm(v_t_text_)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_text_ * scale
            elif cfg_renorm_type == "channel":
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_text_ * scale
            else:
                v_t = v_t_text_
        
        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output
    
    def prepare_audio_features(self, curr_kvlens, curr_rope, audio_features_list, new_token_ids, audio_lengths=None):
        packed_audio_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        packed_audio_features = list()
        newlens, new_rope = list(), list()
        
        # If audio_lengths is not provided, assume full length of features
        if audio_lengths is None:
            audio_lengths = [f.shape[1] for f in audio_features_list]
        
        valid_token_counts = []

        for audio_features, input_len, curr_kvlen, curr_position_id in zip(audio_features_list, audio_lengths, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # packed_text_ids.append(new_token_ids['start_of_audio'])
            # packed_text_indexes.append(_curr)
            # packed_indexes.append(curr)
            # curr += 1
            # _curr += 1

            packed_audio_features.append(audio_features)
            
            # Calculate number of audio tokens based on VALID input length
            # Qwen2-Audio Encoder (Whisper) downsampling factor is 4
            # Formula: ((T + 1) // 2) // 2
            num_audio_tokens = ((input_len + 1) // 2) // 2 
            valid_token_counts.append(num_audio_tokens)

            packed_audio_token_indexes.extend(range(_curr, _curr + num_audio_tokens))
            packed_indexes.extend(range(curr, curr + num_audio_tokens))
            curr += num_audio_tokens
            _curr += num_audio_tokens

            # packed_text_ids.append(new_token_ids['end_of_audio'])
            # packed_text_indexes.append(_curr)
            # packed_indexes.append(curr)
            # curr += 1
            # _curr += 1

            # Use incrementing position IDs for audio tokens
            # Qwen2-Audio uses position_ids that skip the audio tokens? No, it seems to use continuous position ids.
            # But wait, if we look at Qwen2-Audio code, it merges text and audio embeddings.
            # The position ids should correspond to the sequence position.
            # Our packed_position_ids are used for RoPE.
            
            # Let's verify if Qwen2-Audio uses special position ids for audio.
            # In Qwen2-Audio, the audio features replace the <|audio|> token.
            # So the position ids should just increment.
            
            packed_position_ids.extend(range(curr_position_id, curr_position_id + num_audio_tokens))
            
            packed_seqlens.append(num_audio_tokens)
            newlens.append(curr_kvlen + num_audio_tokens)
            # Update rope to reflect the length of audio tokens
            new_rope.append(curr_position_id + num_audio_tokens)

        # Pad audio features
        max_T = max([f.shape[1] for f in packed_audio_features])
        padded_audio_features = torch.zeros(len(packed_audio_features), 128, max_T)
        for i, f in enumerate(packed_audio_features):
            padded_audio_features[i, :, :f.shape[1]] = f

        generation_input = {
            "packed_audio_features": padded_audio_features,
            "packed_audio_und_token_indexes": torch.tensor(packed_audio_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "valid_token_counts": torch.tensor(valid_token_counts, dtype=torch.long),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_audio(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_audio_features: torch.Tensor,
        packed_audio_und_token_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        valid_token_counts: Optional[torch.LongTensor] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # Encode Audio
        # packed_audio_features: [B, 128, T]
        audio_embeds = self.audio_encode(packed_audio_features) # [B, L, D]
        
        # Slice valid tokens if valid_token_counts is provided
        if valid_token_counts is not None:
            valid_embeds_list = []
            for i in range(audio_embeds.shape[0]):
                n = valid_token_counts[i]
                valid_embeds_list.append(audio_embeds[i, :n, :])
            flat_audio_embeds = torch.cat(valid_embeds_list, dim=0)
        else:
            # Fallback to flattening everything (assumes no padding or full usage)
            flat_audio_embeds = audio_embeds.reshape(-1, self.hidden_size)
            
            # Safety check for single sample case
            if flat_audio_embeds.shape[0] > packed_audio_und_token_indexes.shape[0]:
                 flat_audio_embeds = flat_audio_embeds[:packed_audio_und_token_indexes.shape[0]]
        
        packed_sequence[packed_audio_und_token_indexes] = flat_audio_embeds

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    @torch.no_grad()
    def chat_audio(
        self,
        tokenizer,
        new_token_ids,
        audio_processor,
        audios, # List of audio paths or raw waveforms
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # Process Audios
        # We use the processor to get features
        # audios can be paths or arrays
        # processor expects list of arrays or paths?
        # Qwen2Audio processor handles paths if we use 'audios=...'? 
        # Let's assume audios is a list of paths or loaded waveforms.
        # We process them one by one to get features list
        
        audio_features_list = []
        audio_lengths = []
        for audio in audios:
            # Load if path
            if isinstance(audio, str):
                import librosa
                audio, _ = librosa.load(audio, sr=audio_processor.feature_extractor.sampling_rate)
            
            # Use feature_extractor directly to avoid "text input required" error from processor
            # 1. Get unpadded features to know the valid length
            inputs = audio_processor.feature_extractor(audio, return_tensors="pt", sampling_rate=audio_processor.feature_extractor.sampling_rate, padding=False)
            features = inputs.input_features # [1, 128, T]
            valid_len = features.shape[2]
            audio_lengths.append(valid_len)
            
            # 2. Pad to 3000 (Qwen2Audio requirement)
            # Qwen2Audio expects fixed input length of 3000
            padded_features = torch.zeros(1, 128, 3000)
            if valid_len > 3000:
                padded_features[:, :, :] = features[:, :, :3000]
                audio_lengths[-1] = 3000 # Update valid length if truncated
            else:
                padded_features[:, :, :valid_len] = features
            
            audio_features_list.append(padded_features.squeeze(0).to(device))

        # add audios
        if len(audio_features_list) > 0:
            generation_input, newlens, new_rope = self.prepare_audio_features(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                audio_features_list=audio_features_list,
                new_token_ids=new_token_ids,
                audio_lengths=audio_lengths,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_audio(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        # Clean up output
        if '<|im_start|>' in output:
             output = output.split('<|im_start|>')[-1]
        if '<|im_end|>' in output:
             output = output.split('<|im_end|>')[0]

        return output
    
    def audio_get_encoder(self):
        """Get the audio encoder (Whisper)."""
        if not self.config.audio_und:
            raise RuntimeError("Audio understanding is not enabled in config")
        return self.audio_encoder
    
    def audio_encode(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features through encoder and projector.
        
        Args:
            audio_features: Either:
                - Raw mel spectrogram features [batch, mel_bins, time_frames] for audio encoder
                - Already encoded features [seq_len, 1280] (encoder output)
            
        Returns:
            projected_features: [seq_len, hidden_size] audio embeddings ready for LLM
        """
        if not self.config.audio_und:
            raise RuntimeError("Audio understanding is not enabled in config")
            
        # Check if audio_features are already encoded
        is_already_encoded = False
        if audio_features.dim() == 2:
            if audio_features.shape[-1] == self.audio_encoder_hidden_size:
                is_already_encoded = True
        
        # Pass through audio encoder only if not already encoded
        if self.audio_encoder is not None and not is_already_encoded:
            audio_hidden_states = self.audio_encoder(audio_features)
            # Extract hidden states if it's a model output object
            if hasattr(audio_hidden_states, 'last_hidden_state'):
                audio_hidden_states = audio_hidden_states.last_hidden_state
            # Remove batch dimension if present
            # if audio_hidden_states.dim() == 3:
            #     audio_hidden_states = audio_hidden_states.squeeze(0)
        else:
            # Already encoded or no encoder available
            audio_hidden_states = audio_features
        
        # Project to LLM hidden size
        projected_features = self.audio_projector(audio_hidden_states)
        
        return projected_features
    
    def audio_forward(
        self,
        # Audio inputs
        audio_features: Optional[torch.Tensor] = None,
        audio_token_indexes: Optional[torch.LongTensor] = None,
        
        # Text inputs
        input_ids: Optional[torch.LongTensor] = None,
        text_token_indexes: Optional[torch.LongTensor] = None,
        
        # Sequence info
        sequence_length: int = None,
        sample_lens: List[int] = None,
        attention_mask = None,
        position_ids: Optional[torch.LongTensor] = None,
        
        # Labels for training
        labels: Optional[torch.LongTensor] = None,
        
        # Mode control
        mode: str = "und",  # "und" for understanding, "gen" for generation
        
        **kwargs
    ):
        """
        Forward pass for audio-text multimodal processing.
        
        Args:
            audio_features: Audio input features [batch, audio_len, ...]
            audio_token_indexes: Positions where audio tokens go in packed sequence
            input_ids: Text token IDs
            text_token_indexes: Positions where text tokens go in packed sequence
            sequence_length: Total length of packed sequence
            sample_lens: List of per-sample lengths
            mode: "und" for understanding, "gen" for generation
        """
        if not self.config.audio_und:
            raise RuntimeError("Audio understanding is not enabled in config")
            
        # Build packed sequence
        packed_sequence = self.language_model.model.embed_tokens(input_ids).new_zeros(
            size=(sequence_length, self.hidden_size)
        )
        
        # Add text embeddings
        if input_ids is not None and text_token_indexes is not None:
            text_embeddings = self.language_model.model.embed_tokens(input_ids)
            packed_sequence[text_token_indexes] = text_embeddings
        
        # Add audio embeddings
        if audio_features is not None and audio_token_indexes is not None:
            audio_embeddings = self.audio_encode(audio_features)
            if audio_embeddings.dim() == 3:
                audio_embeddings = audio_embeddings.reshape(-1, self.hidden_size)
            packed_sequence[audio_token_indexes] = audio_embeddings
        
        # Define understanding and generation token indexes
        packed_und_token_indexes = text_token_indexes
        if audio_token_indexes is not None:
            packed_und_token_indexes = torch.cat([text_token_indexes, audio_token_indexes], dim=0)
        
        # For training, generation tokens would be the label positions
        packed_gen_token_indexes = None
        if labels is not None:
            # Implementation depends on your sequence plan
            pass
        
        # Forward through language model
        if self.training:
            outputs = self.language_model.forward_train(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=position_ids,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
        else:
            # Generate packed_query_indexes
            packed_query_indexes = torch.arange(sequence_length, dtype=torch.long)
            if hasattr(packed_sequence, 'device'):
                packed_query_indexes = packed_query_indexes.to(packed_sequence.device)
            
            outputs = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=torch.tensor(sample_lens, dtype=torch.int) if sample_lens else None,
                packed_query_position_ids=position_ids,
                packed_query_indexes=packed_query_indexes,
                mode=mode,
                **kwargs
            )
        
        return outputs
    
    @classmethod
    def audio_from_pretrained_qwen2_audio(cls, existing_bagel_model, qwen2_audio_model_name: str, device_map="auto", **kwargs):
        """
        Load Qwen2-Audio weights into an existing Bagel model for audio understanding.
        
        This method:
        1. Loads Qwen2-Audio's audio encoder and projector
        2. Loads Qwen2-Audio's LLM base weights into understanding experts
        3. Duplicates weights to initialize MoE generation experts
        
        Args:
            existing_bagel_model: An existing Bagel model instance
            qwen2_audio_model_name: HuggingFace model name (e.g., "Qwen/Qwen2-Audio-7B-Instruct")
            device_map: Device placement strategy (default: "auto")
            **kwargs: Additional arguments for model loading
        
        Returns:
            The same Bagel model with audio components loaded
        """
        print(f"🔄 Loading Qwen2-Audio components into Bagel: {qwen2_audio_model_name}")
        
        if not existing_bagel_model.config.audio_und:
            raise RuntimeError("Audio understanding must be enabled in BagelConfig")
        
        try:
            from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig
        except ImportError:
            raise ImportError(
                "Qwen2-Audio is not available. "
                "Please install: pip install transformers>=4.37.0"
            )
        
        # Step 1: Load pretrained Qwen2-Audio
        print("📋 Loading Qwen2-Audio configuration...")
        pretrained_config = Qwen2AudioConfig.from_pretrained(qwen2_audio_model_name)
        
        print("⬇️  Loading pretrained Qwen2-Audio model...")
        pretrained_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            qwen2_audio_model_name,
            device_map=device_map,
            torch_dtype=kwargs.get('torch_dtype', torch.bfloat16),
            trust_remote_code=True
        )
        
        # Step 2: Load audio_encoder
        existing_bagel_model.audio_encoder = pretrained_model.audio_tower
        print("  ✅ Audio encoder loaded")
        
        # Step 3: Map and load weights
        print("🔄 Mapping weights to Bagel architecture...")
        existing_bagel_model._audio_load_pretrained_weights(pretrained_model, pretrained_config)
        
        # Step 4: Initialize MoE generation experts
        print("🔀 Initializing MoE generation experts...")
        existing_bagel_model._audio_initialize_moe_experts()
        
        # Step 5: Ensure correct device and dtype
        print("🔧 Ensuring modules are on correct device...")
        target_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        
        if torch.cuda.is_available():
            target_device = torch.device("cuda:0")
            print(f"  🎯 Using device: {target_device}")
            
            # Move audio_projector
            if hasattr(existing_bagel_model, 'audio_projector'):
                existing_bagel_model.audio_projector = existing_bagel_model.audio_projector.to(
                    device=target_device, dtype=target_dtype
                )
                print(f"  ✅ audio_projector on {target_device}")
        
        print("✅ Qwen2-Audio components loaded successfully!")
        return existing_bagel_model
    
    def _audio_load_pretrained_weights(self, pretrained_model, pretrained_config):
        """
        Load weights from pretrained Qwen2-Audio model to Bagel.
        
        This handles:
        - Audio projector weights
        - LLM base weights (understanding expert)
        """
        print("  📦 Loading Audio Projector weights...")
        pretrained_proj_state = pretrained_model.multi_modal_projector.state_dict()
        
        # Handle nested structure
        if 'linear.weight' in pretrained_proj_state:
            our_proj_state = {
                'weight': pretrained_proj_state['linear.weight'],
                'bias': pretrained_proj_state['linear.bias']
            }
        else:
            our_proj_state = pretrained_proj_state
        
        self.audio_projector.load_state_dict(our_proj_state, strict=True)
        
        print("  📦 Loading Language Model base weights...")
        pretrained_lm = pretrained_model.language_model
        
        # Load embeddings
        # Handle size mismatch: Qwen2-Audio vocab size might be larger than ours
        pretrained_embed = pretrained_lm.model.embed_tokens.state_dict()
        current_vocab_size = self.language_model.model.embed_tokens.weight.shape[0]
        pretrained_vocab_size = pretrained_embed['weight'].shape[0]
        
        if current_vocab_size != pretrained_vocab_size:
            print(f"  ⚠️ Vocab size mismatch: Current {current_vocab_size} vs Pretrained {pretrained_vocab_size}")
            if current_vocab_size < pretrained_vocab_size:
                print(f"  ✂️ Truncating pretrained embeddings to {current_vocab_size}")
                pretrained_embed['weight'] = pretrained_embed['weight'][:current_vocab_size, :]
            else:
                print(f"  ➕ Extending pretrained embeddings (random init for new tokens)")
                # Keep original weights, new ones are already random init
                # We only load the overlapping part
                new_weight = self.language_model.model.embed_tokens.weight.data.clone()
                new_weight[:pretrained_vocab_size, :] = pretrained_embed['weight']
                pretrained_embed['weight'] = new_weight

        self.language_model.model.embed_tokens.load_state_dict(
            pretrained_embed,
            strict=True
        )
        
        # Load norm layers
        self.language_model.model.norm.load_state_dict(
            pretrained_lm.model.norm.state_dict(),
            strict=True
        )
        
        # Load lm_head
        if hasattr(pretrained_lm, 'lm_head'):
            self.language_model.lm_head.load_state_dict(
                pretrained_lm.lm_head.state_dict(),
                strict=True
            )
        
        # Load decoder layers (understanding expert only)
        print("  📦 Loading Decoder Layers (understanding expert)...")
        for layer_idx in range(len(self.language_model.model.layers)):
            pretrained_layer = pretrained_lm.model.layers[layer_idx]
            our_layer = self.language_model.model.layers[layer_idx]
            
            # Load attention weights (understanding path)
            self._audio_load_attention_weights(
                our_layer.self_attn,
                pretrained_layer.self_attn,
                mode='understanding'
            )
            
            # Load MLP weights (understanding expert)
            our_layer.mlp.load_state_dict(
                pretrained_layer.mlp.state_dict(),
                strict=True
            )
            
            # Load layer norms
            our_layer.input_layernorm.load_state_dict(
                pretrained_layer.input_layernorm.state_dict(),
                strict=True
            )
            our_layer.post_attention_layernorm.load_state_dict(
                pretrained_layer.post_attention_layernorm.state_dict(),
                strict=True
            )
            
        print("  ✅ Base weights loaded successfully!")
    
    def _audio_load_attention_weights(self, our_attn, pretrained_attn, mode='understanding'):
        """
        Load attention weights from pretrained model.
        
        For understanding mode:
        - Load to q_proj, k_proj, v_proj, o_proj
        """
        if mode == 'understanding':
            # Load Q/K/V projections
            our_attn.q_proj.load_state_dict(pretrained_attn.q_proj.state_dict(), strict=True)
            our_attn.k_proj.load_state_dict(pretrained_attn.k_proj.state_dict(), strict=True)
            our_attn.v_proj.load_state_dict(pretrained_attn.v_proj.state_dict(), strict=True)
            our_attn.o_proj.load_state_dict(pretrained_attn.o_proj.state_dict(), strict=True)
            
            # Keep random initialization for q_norm, k_norm if present
            pass
    
    def _audio_initialize_moe_experts(self):
        """
        Initialize MoE generation experts by copying weights from understanding experts.
        """
        print("  🔀 Copying understanding expert weights to generation experts...")
        
        # Check if we are using a Slim Expert (Bottleneck)
        is_slim_expert = False
        if hasattr(self.language_model.config, "gen_intermediate_size") and self.language_model.config.gen_intermediate_size is not None:
            if self.language_model.config.gen_intermediate_size != self.language_model.config.intermediate_size:
                is_slim_expert = True
                print(f"  ⚠️  Detected Slim Expert (Bottleneck): Gen Size {self.language_model.config.gen_intermediate_size} vs Und Size {self.language_model.config.intermediate_size}")
        
        # Copy model-level norm for generation
        if hasattr(self.language_model.model, 'norm_moe_gen'):
            with torch.no_grad():
                for param_name, param in self.language_model.model.norm.named_parameters():
                    target_param = getattr(self.language_model.model.norm_moe_gen, param_name)
                    target_param.data.copy_(param.data)
        
        # Copy layer-by-layer
        for layer in self.language_model.model.layers:
            # Handle Slim Expert Replacement
            if is_slim_expert:
                # Create new config for generation expert
                gen_config = copy.deepcopy(self.language_model.config)
                gen_config.intermediate_size = self.language_model.config.gen_intermediate_size
                # Replace the module
                layer.mlp_moe_gen = Qwen2MLP(gen_config).to(layer.mlp.weight.device if hasattr(layer.mlp, 'weight') else next(layer.mlp.parameters()).device).to(dtype=next(layer.mlp.parameters()).dtype)
                # Initialize weights (randomly, as we can't copy from larger/smaller MLP)
                # Qwen2MLP init already does random init.
            
            # Copy attention generation projections
            with torch.no_grad():
                layer.self_attn.q_proj_moe_gen.weight.data.copy_(layer.self_attn.q_proj.weight.data)
                layer.self_attn.q_proj_moe_gen.bias.data.copy_(layer.self_attn.q_proj.bias.data)
                
                layer.self_attn.k_proj_moe_gen.weight.data.copy_(layer.self_attn.k_proj.weight.data)
                layer.self_attn.k_proj_moe_gen.bias.data.copy_(layer.self_attn.k_proj.bias.data)
                
                layer.self_attn.v_proj_moe_gen.weight.data.copy_(layer.self_attn.v_proj.weight.data)
                layer.self_attn.v_proj_moe_gen.bias.data.copy_(layer.self_attn.v_proj.bias.data)
                
                layer.self_attn.o_proj_moe_gen.weight.data.copy_(layer.self_attn.o_proj.weight.data)
            
            # Copy QK norms for generation if present
            if hasattr(layer.self_attn, 'q_norm_moe_gen') and not isinstance(layer.self_attn.q_norm_moe_gen, nn.Identity):
                with torch.no_grad():
                    for param_name, param in layer.self_attn.q_norm.named_parameters():
                        target_param = getattr(layer.self_attn.q_norm_moe_gen, param_name)
                        target_param.data.copy_(param.data)
                    
                    for param_name, param in layer.self_attn.k_norm.named_parameters():
                        target_param = getattr(layer.self_attn.k_norm_moe_gen, param_name)
                        target_param.data.copy_(param.data)
            
            # Copy MLP generation expert ONLY if NOT slim expert
            if not is_slim_expert:
                with torch.no_grad():
                    mlp_state = layer.mlp.state_dict()
                    layer.mlp_moe_gen.load_state_dict(mlp_state, strict=True)
            
            # Copy layer norms for generation path
            with torch.no_grad():
                for param_name, param in layer.post_attention_layernorm.named_parameters():
                    target_param = getattr(layer.post_attention_layernorm_moe_gen, param_name)
                    target_param.data.copy_(param.data)
        
        print("  ✅ MoE experts initialized!")
    
    def audio_load_qwen2_reference_model(self, model_name: str = "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"):
        """
        Load a reference Qwen2-Audio model for generation comparison/fallback.
        
        Args:
            model_name: HuggingFace model name for Qwen2-Audio
            device_map: Device placement strategy
        """
        if not self.config.audio_und:
            raise RuntimeError("Audio understanding is not enabled in config")
            
        try:
            from transformers import Qwen2AudioForConditionalGeneration
        except ImportError:
            raise ImportError("Qwen2-Audio not available in transformers")
        
        print(f"📥 Loading Qwen2-Audio reference model: {model_name}")
        self._audio_qwen2_ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self._audio_qwen2_ref_model.eval()
        print("✅ Reference model loaded successfully!")
    
    def audio_generate_with_qwen2_reference(
        self,
        audio_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_text: Optional[str] = None,
        raw_audio: Optional[Any] = None,  # numpy array
        processor=None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        tokenizer=None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text using Qwen2-Audio's native generate() method as reference.
        
        Args:
            audio_features: Raw mel spectrogram features or None
            input_ids: Text input token IDs or None
            input_text: Text input string (requires processor)
            raw_audio: Raw audio waveform as numpy array (requires processor)
            processor: Qwen2AudioProcessor instance
            max_new_tokens: Maximum tokens to generate
            temperature, top_k, top_p, do_sample: Generation parameters
            tokenizer: Tokenizer
            
        Returns:
            Generated token IDs
        """
        if not self.config.audio_und:
            raise RuntimeError("Audio understanding is not enabled in config")
            
        if self._audio_qwen2_ref_model is None:
            raise RuntimeError(
                "Reference model not loaded. Call audio_load_qwen2_reference_model() first."
            )
        
        device = next(self._audio_qwen2_ref_model.parameters()).device
        
        # Best approach: Use processor if available
        if processor is not None:
            # Use processor to format inputs properly
            if input_text is None and input_ids is not None and tokenizer is not None:
                input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            
            if input_text is None:
                raise ValueError("Either input_text or input_ids must be provided")
            
            # Prepare audio
            audio_list = None
            if raw_audio is not None:
                import numpy as np
                audio_list = [raw_audio] if isinstance(raw_audio, np.ndarray) else raw_audio
            elif audio_features is not None:
                print("    ⚠️  Warning: audio_features provided but processor needs raw_audio")
                audio_list = None
            
            # Use processor to format inputs
            try:
                inputs = processor(text=input_text, audios=audio_list, return_tensors="pt", padding=True)
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(device)
                
                generated_ids = self._audio_qwen2_ref_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    **kwargs
                )
                
                return generated_ids.squeeze(0) if generated_ids.dim() > 1 else generated_ids
            except Exception as e:
                print(f"    ⚠️  Error using processor: {e}")
                raise
        
        # Fallback: Manual preparation (not recommended)
        raise NotImplementedError("Manual input preparation not implemented. Please use processor.")
    
   