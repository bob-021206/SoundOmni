# Copyright 2025
# Model configuration presets for different model sizes

from ..models.qwen2_audio_bagel import Qwen2AudioBagelConfig


# Configuration presets matching Qwen2-Audio model sizes
PRESET_CONFIGS = {
    "qwen2-audio-7b": {
        "vocab_size": 151936,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "qk_norm": True,
        "use_moe": True,
        "freeze_und": False,
        "audio_encoder_name": "openai/whisper-large-v3",
    },
    
    "qwen2-audio-bagel-small": {
        # Smaller config for testing
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "qk_norm": True,
        "use_moe": True,
        "freeze_und": False,
        "audio_encoder_name": "openai/whisper-base",
    },
}


def get_config(preset: str = "qwen2-audio-7b", **overrides) -> Qwen2AudioBagelConfig:
    """
    Get a model configuration by preset name, with optional overrides.
    
    Args:
        preset: Name of preset configuration
        **overrides: Additional config parameters to override
    
    Returns:
        Qwen2AudioBagelConfig instance
    
    Example:
        # Get default 7B config
        config = get_config("qwen2-audio-7b")
        
        # Get 7B config with custom settings
        config = get_config(
            "qwen2-audio-7b",
            num_hidden_layers=16,  # Reduce layers for faster training
            freeze_und=True,        # Freeze understanding path
        )
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(PRESET_CONFIGS.keys())}"
        )
    
    config_dict = PRESET_CONFIGS[preset].copy()
    config_dict.update(overrides)
    
    return Qwen2AudioBagelConfig(**config_dict)

