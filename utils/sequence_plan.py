# Copyright 2025
# Sequence Plan utilities for defining understanding vs generation tokens
# This is crucial for MoT routing: which tokens go through which expert path

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class SequencePlan:
    """
    Defines the structure of a packed multimodal sequence.
    
    Attributes:
        sequence_length: Total number of tokens in packed sequence
        sample_lens: List of lengths for each sample in batch
        
        # Understanding tokens (audio + input text)
        audio_token_indexes: Where audio embeddings are placed
        input_text_token_indexes: Where input text tokens are placed
        packed_und_token_indexes: Combined understanding token indexes
        
        # Generation tokens (output text, future: output audio)
        output_text_token_indexes: Where output text tokens are placed
        packed_gen_token_indexes: Combined generation token indexes
        
        # Position IDs for RoPE
        position_ids: Position ID for each token in sequence
    """
    sequence_length: int
    sample_lens: List[int]
    
    # Understanding
    audio_token_indexes: Optional[torch.LongTensor] = None
    input_text_token_indexes: Optional[torch.LongTensor] = None
    packed_und_token_indexes: Optional[torch.LongTensor] = None
    
    # Generation
    output_text_token_indexes: Optional[torch.LongTensor] = None
    packed_gen_token_indexes: Optional[torch.LongTensor] = None
    
    # Positions
    position_ids: Optional[torch.LongTensor] = None
    
    def to(self, device):
        """Move all tensors to device."""
        for attr in ['audio_token_indexes', 'input_text_token_indexes', 
                     'packed_und_token_indexes', 'output_text_token_indexes',
                     'packed_gen_token_indexes', 'position_ids']:
            value = getattr(self, attr, None)
            if value is not None and isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
        return self


class SequencePlanBuilder:
    """
    Builder class for constructing sequence plans for audio-text multimodal data.
    
    Example usage:
        builder = SequencePlanBuilder()
        builder.add_audio_segment(start=0, length=100)  # Audio tokens 0-99
        builder.add_input_text(start=100, length=20)    # Input text tokens 100-119
        builder.add_output_text(start=120, length=30)   # Output text tokens 120-149
        plan = builder.build()
    """
    
    def __init__(self):
        self.audio_indexes = []
        self.input_text_indexes = []
        self.output_text_indexes = []
        self.position_ids = []
        self.sample_lens = []
        self.current_length = 0
    
    def add_audio_segment(self, length: int, position_offset: int = 0):
        """
        Add audio token indexes to the sequence.
        
        Args:
            length: Number of audio tokens
            position_offset: Starting position ID for this segment
        """
        start = self.current_length
        end = start + length
        self.audio_indexes.extend(range(start, end))
        self.position_ids.extend(range(position_offset, position_offset + length))
        self.current_length = end
        return self
    
    def add_input_text(self, token_ids: List[int], position_offset: int = 0):
        """
        Add input text token indexes to the sequence.
        
        Args:
            token_ids: List of input token IDs (just for length, actual IDs handled separately)
            position_offset: Starting position ID for this segment
        """
        start = self.current_length
        length = len(token_ids)
        end = start + length
        self.input_text_indexes.extend(range(start, end))
        self.position_ids.extend(range(position_offset, position_offset + length))
        self.current_length = end
        return self
    
    def add_output_text(self, length: int, position_offset: int = 0):
        """
        Add output text token indexes to the sequence.
        
        Args:
            length: Number of output tokens
            position_offset: Starting position ID for this segment
        """
        start = self.current_length
        end = start + length
        self.output_text_indexes.extend(range(start, end))
        self.position_ids.extend(range(position_offset, position_offset + length))
        self.current_length = end
        return self
    
    def add_sample_boundary(self):
        """Mark the end of current sample and start a new one."""
        self.sample_lens.append(self.current_length - sum(self.sample_lens))
        return self
    
    def build(self) -> SequencePlan:
        """Build and return the final SequencePlan."""
        # Finalize sample lens
        if sum(self.sample_lens) < self.current_length:
            self.sample_lens.append(self.current_length - sum(self.sample_lens))
        
        # Convert to tensors
        audio_token_indexes = (
            torch.tensor(self.audio_indexes, dtype=torch.long) 
            if self.audio_indexes else None
        )
        input_text_token_indexes = (
            torch.tensor(self.input_text_indexes, dtype=torch.long) 
            if self.input_text_indexes else None
        )
        output_text_token_indexes = (
            torch.tensor(self.output_text_indexes, dtype=torch.long) 
            if self.output_text_indexes else None
        )
        
        # Combine understanding indexes (audio + input text)
        und_indexes_list = []
        if audio_token_indexes is not None:
            und_indexes_list.append(audio_token_indexes)
        if input_text_token_indexes is not None:
            und_indexes_list.append(input_text_token_indexes)
        packed_und_token_indexes = (
            torch.cat(und_indexes_list, dim=0) if und_indexes_list else None
        )
        
        # Generation indexes (output text, future: output audio)
        packed_gen_token_indexes = output_text_token_indexes
        
        position_ids = torch.tensor(self.position_ids, dtype=torch.long)
        
        return SequencePlan(
            sequence_length=self.current_length,
            sample_lens=self.sample_lens,
            audio_token_indexes=audio_token_indexes,
            input_text_token_indexes=input_text_token_indexes,
            packed_und_token_indexes=packed_und_token_indexes,
            output_text_token_indexes=output_text_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            position_ids=position_ids,
        )


def create_sequence_plan(
    audio_lengths: Optional[List[int]] = None,
    input_text_lengths: Optional[List[int]] = None,
    output_text_lengths: Optional[List[int]] = None,
    batch_size: int = 1,
) -> SequencePlan:
    """
    Helper function to create a sequence plan for a batch.
    
    Args:
        audio_lengths: List of audio sequence lengths for each sample
        input_text_lengths: List of input text lengths for each sample
        output_text_lengths: List of output text lengths for each sample
        batch_size: Number of samples in batch
    
    Returns:
        SequencePlan for the batch
    
    Example:
        # Single sample: 100 audio tokens + 20 input text + 30 output text
        plan = create_sequence_plan(
            audio_lengths=[100],
            input_text_lengths=[20],
            output_text_lengths=[30],
            batch_size=1
        )
    """
    if audio_lengths is None:
        audio_lengths = [0] * batch_size
    if input_text_lengths is None:
        input_text_lengths = [0] * batch_size
    if output_text_lengths is None:
        output_text_lengths = [0] * batch_size
    
    builder = SequencePlanBuilder()
    position_offset = 0
    
    for i in range(batch_size):
        # Add audio
        if audio_lengths[i] > 0:
            builder.add_audio_segment(audio_lengths[i], position_offset)
            position_offset += 1  # Audio takes 1 position ID
        
        # Add input text
        if input_text_lengths[i] > 0:
            builder.add_input_text(
                token_ids=list(range(input_text_lengths[i])),  # Dummy IDs
                position_offset=position_offset
            )
            position_offset += input_text_lengths[i]
        
        # Add output text
        if output_text_lengths[i] > 0:
            builder.add_output_text(output_text_lengths[i], position_offset)
            position_offset += output_text_lengths[i]
        
        # Mark sample boundary
        builder.add_sample_boundary()
    
    return builder.build()


def visualize_sequence_plan(plan: SequencePlan):
    """
    Visualize a sequence plan (useful for debugging).
    
    Args:
        plan: SequencePlan to visualize
    """
    print(f"Sequence Plan Visualization")
    print(f"{'='*60}")
    print(f"Total Length: {plan.sequence_length}")
    print(f"Sample Lengths: {plan.sample_lens}")
    print(f"\nUnderstanding Tokens (Audio + Input Text):")
    if plan.audio_token_indexes is not None:
        print(f"  Audio: {plan.audio_token_indexes.tolist()[:10]}{'...' if len(plan.audio_token_indexes) > 10 else ''}")
    if plan.input_text_token_indexes is not None:
        print(f"  Input Text: {plan.input_text_token_indexes.tolist()[:10]}{'...' if len(plan.input_text_token_indexes) > 10 else ''}")
    
    print(f"\nGeneration Tokens (Output Text):")
    if plan.output_text_token_indexes is not None:
        print(f"  Output Text: {plan.output_text_token_indexes.tolist()[:10]}{'...' if len(plan.output_text_token_indexes) > 10 else ''}")
    
    print(f"\nPosition IDs (first 20): {plan.position_ids.tolist()[:20]}")
    print(f"{'='*60}\n")

