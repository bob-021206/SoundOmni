# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .t2a_dataset import T2AIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    't2a_pretrain': T2AIterableDataset,  # Text-to-Audio dataset
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    't2a_pretrain': {
        'audiocaps': {
            'csv_base_dir': 'dataset2.0',  # CSV files base directory (relative to text_audio/)
            'audio_dir': '/mnt/cfs/5vr0p6/yimingjing/data/audiocaps_raw_audio',  # WAV files directory (relative to text_audio/)
            'num_files': 1,  # number of CSV files
            # num_total_samples will be determined by split
            'num_total_samples': {
                'train': 91256,
                'val': 2475,
                'test': 4875,
            },
            'default_split': 'train',  # default split if not specified in YAML
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}