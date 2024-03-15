from dataclasses import dataclass
from typing import List, Type, Optional, Union, Dict

from jsonargparse import ArgumentParser, ActionConfigFile

@dataclass
class ModelConfig:
    model_id: str

@dataclass
class DataConfig:
    dataset_name: str
    split: str
    total_sources: int
    random_sample: bool

@dataclass
class GeneralConfig:
    batch_size: int
    experiment_key: str

def parse_args():
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_class_arguments(DataConfig, 'data_config')
    parser.add_class_arguments(ModelConfig, 'model_config')
    parser.add_class_arguments(GeneralConfig, 'general_config')
    
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument('--mode', choices=['run_all', 'run_prob', 'run_pid', 'extract_only'], default='run_all')

    return parser.parse_args()