# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import random
import traceback
from pathlib import Path

import torch

from .distributed_iterable_dataset import DistributedIterableDataset


class T2AIterableDataset(DistributedIterableDataset):
    """
    Text-to-Audio Dataset
    
    Reads audio descriptions from CSV files and loads corresponding WAV audio files from dataset folder
    
    CSV format:
        audiocap_id,youtube_id,start_time,caption
        
    Audio file naming:
        {youtube_id}_{start_time}.wav
    """
    def __init__(
        self,
        dataset_name,
        audio_transform,
        tokenizer,
        csv_path_list,
        audio_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        split='train',  # 'train', 'val', or 'test'
        **kwargs,
    ):
        """
        Args:
            dataset_name: Name of the dataset
            audio_transform: Audio processor (AudioTransform), returns raw waveform [C, T]
            tokenizer: Text tokenizer
            csv_path_list: List of CSV file paths (supports multiple datasets)
            audio_dir_list: List of audio folder paths
            num_used_data: List of number of samples to use per dataset
            local_rank: Current GPU rank
            world_size: Total number of GPUs
            num_workers: Number of DataLoader workers
            data_status: Resume training status
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.audio_transform = audio_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.split = split
        
        # Build data paths
        self.data_paths = self.get_data_paths(
            csv_path_list, 
            audio_dir_list, 
            num_used_data
        )
        self.set_epoch()

    def get_data_paths(self, csv_path_list, audio_dir_list, num_used_data):
        """
        Read CSV files and build (csv_path, audio_dir, row_data) tuple list
        
        Args:
            csv_path_list: List of CSV file paths
            audio_dir_list: List of audio directories
            num_used_data: List of number of data to use
            
        Returns:
            data_paths: [(csv_idx, row_idx, row_data, audio_dir), ...]
        """
        data_paths = []
        
        for csv_idx, (csv_path, audio_dir, num_data) in enumerate(
            zip(csv_path_list, audio_dir_list, num_used_data)
        ):
            # Read CSV file
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Limit the amount of data to use
            if num_data > 0 and num_data < len(rows):
                rows = rows[:num_data]
            
            # Build data paths
            for row_idx, row in enumerate(rows):
                data_paths.append((csv_idx, row_idx, row, audio_dir))
            
            if self.local_rank == 0:
                print(f"Loaded {len(rows)} samples from {csv_path}")
        
        return data_paths

    def __iter__(self):
        """
        Iterate over the dataset, yielding processed samples
        """
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # Resume training from checkpoint
        start_idx = 0
        if self.data_status is not None and worker_id in self.data_status:
            status = self.data_status[worker_id]
            if isinstance(status, int):
                start_idx = status + 1
            elif isinstance(status, list):
                # Compatibility with old checkpoints that saved [csv_idx, row_idx]
                target_csv_idx, target_row_idx = status
                found = False
                for i, (csv_idx, row_idx, _, _) in enumerate(data_paths_per_worker):
                    if csv_idx == target_csv_idx and row_idx == target_row_idx:
                        start_idx = i + 1
                        found = True
                        break
                if not found:
                    print(f"Warning: Could not find resume sample {status} in worker {worker_id}. Starting from 0.")
            else:
                print(f"Warning: Unknown status type {type(status)} for worker {worker_id}. Starting from 0.")

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at sample#{start_idx}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[start_idx:]
            
            for sample_idx, (csv_idx, row_idx, row, audio_dir) in enumerate(
                data_paths_per_worker_, start=start_idx
            ):
                try:
                    # 1. Parse CSV row data
                    audiocap_id = row['audiocap_id']
                    youtube_id = row['youtube_id']
                    start_time = row['start_time']
                    caption = row['caption']
                    
                    # 2. Build audio file path
                    audio_filename = f"{youtube_id}_{start_time}.wav"
                    audio_path = os.path.join(audio_dir, audio_filename)
                    
                    # Check if audio file exists
                    if not os.path.exists(audio_path):
                        print(f"Warning: Audio file not found: {audio_path}")
                        continue
                    
                    # 3. Load and process audio
                    try:
                        audio_waveform = self.audio_transform(audio_path)  # [C, T]
                    except Exception as e:
                        print(f"Error processing audio {audio_path}: {e}")
                        traceback.print_exc()
                        continue
                    
                    # 4. Tokenize caption
                    try:
                        caption_token = self.tokenizer.encode(caption)
                    except Exception as e:
                        print(f"Error tokenizing caption '{caption}': {e}")
                        continue
                    
                    if len(caption_token) == 0:
                        print(f"Warning: Empty caption token for {audiocap_id}")
                        caption_token = self.tokenizer.encode(' ')
                    
                    # 5. Calculate token count
                    # Note: Raw waveform shape = [C, T], e.g. [2, 441000]
                    # - C = 2 (stereo)
                    # - T = 441000 (44.1kHz × 10 seconds)
                    # VAE downsamples by 2048 (441000 / 2048 = 215)
                    num_audio_tokens = audio_waveform.shape[-1] // 2048
                    if num_audio_tokens == 0:
                        num_audio_tokens = 1
                    
                    num_text_tokens = len(caption_token)
                    num_tokens = num_text_tokens + num_audio_tokens
                    
                    # 6. Build sequence_plan
                    sequence_plan = [
                        # Part 1: Input text (caption)
                        {
                            'type': 'text',
                            'enable_cfg': 1,  # Can be used for CFG
                            'loss': 0,  # No loss computation (input)
                            'special_token_loss': 0,
                            'special_token_label': None,
                        },
                        # Part 2: Target audio (audio latent)
                        {
                            'type': 'audio_latent',  # New type
                            'enable_cfg': 0,  # Not used for CFG
                            'loss': 1,  # Compute MSE loss (output)
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                    ]
                    
                    # 7. Build sample dictionary
                    sample = {
                        'audio_feature_list': [audio_waveform],  # [C, T] raw waveform
                        'text_ids_list': [caption_token],
                        'num_tokens': num_tokens,
                        'sequence_plan': sequence_plan,
                        'data_indexes': {
                            'data_indexes': sample_idx,
                            'worker_id': worker_id,
                            'dataset_name': self.dataset_name,
                            'audiocap_id': audiocap_id,
                            'audio_file': audio_filename,
                        }
                    }
                    
                    yield sample
                    
                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    traceback.print_exc()
                    continue
            
            # Repeat dataset
            start_idx = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class T2ASplitDataset:
    """
    Helper class: Select the correct CSV file based on split parameter
    
    Usage:
        dataset = T2ASplitDataset(
            base_dir='path/to/dataset2.0',
            split='train'  # or 'val' or 'test'
        )
    """
    def __init__(self, base_dir, split='train'):
        self.base_dir = Path(base_dir)
        self.split = split
        
        # CSV file paths
        self.csv_files = {
            'train': self.base_dir / 'train.csv',
            'val': self.base_dir / 'val.csv',
            'test': self.base_dir / 'test.csv',
        }
        
        if split not in self.csv_files:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.csv_files.keys())}")
    
    def get_csv_path(self):
        """Return the CSV file path for the current split"""
        csv_path = self.csv_files[self.split]
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return str(csv_path)
