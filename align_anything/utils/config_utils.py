# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Hydra-based configuration management for align-anything."""

import os
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


class ConfigManager:
    """Hydra configuration manager for align-anything."""
    
    def __init__(self):
        self.config_dir = self._get_config_dir()
        
    def _get_config_dir(self) -> str:
        """Get the absolute path to the config directory."""
        current_file_path = os.path.abspath(__file__)
        parent_path = os.path.dirname(os.path.dirname(current_file_path))
        return os.path.join(parent_path, 'configs', 'train', 'text_to_text')
    
    def load_config(self, config_name: str, overrides: Optional[list] = None) -> DictConfig:
        """
        Load configuration using Hydra with inheritance support.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            overrides: List of override strings (e.g., ['model_path=/path/to/model'])
            
        Returns:
            DictConfig: Loaded and resolved configuration
        """
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize Hydra with the config directory
        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            # Compose configuration with overrides
            cfg = compose(config_name=config_name, overrides=overrides or [])
            
        return cfg
    
    def load_multi_turn_config(self, 
                              config_name: str = "ppo_multi_turn",
                              model_path: str = None,
                              **kwargs) -> DictConfig:
        """
        Load multi-turn PPO configuration with common overrides.
        
        Args:
            config_name: Name of the config file
            model_path: Path to the model
            **kwargs: Additional overrides as key=value pairs
            
        Returns:
            DictConfig: Loaded configuration
        """
        overrides = []
        
        # Add model path override if provided
        if model_path:
            overrides.append(f"model_path={model_path}")
            
        # Add any additional overrides from kwargs
        for key, value in kwargs.items():
            overrides.append(f"{key}={value}")
            
        return self.load_config(config_name, overrides)


def read_hydra_cfgs(config_name: str, 
                   mode: str = "train", 
                   task: str = "text_to_text",
                   overrides: Optional[list] = None) -> tuple[DictConfig, dict]:
    """
    Read configuration using Hydra and also load DeepSpeed config.
    
    Args:
        config_name: Name of the config file (e.g., "ppo_multi_turn")
        mode: Training mode (default: "train")
        task: Task type (default: "text_to_text")
        overrides: List of override strings
        
    Returns:
        tuple: (hydra_config, deepspeed_config)
    """
    # Get config directory
    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    config_dir = os.path.join(parent_path, 'configs', mode, task)
    
    # Load Hydra config
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    
    # Load DeepSpeed config if specified
    ds_cfgs = None
    if 'train_cfgs' in cfg and 'ds_cfgs' in cfg.train_cfgs:
        zero_stage_file = os.getenv('ZERO_STAGE_FILE', cfg.train_cfgs.ds_cfgs)
        ds_cfgs_path = os.path.join(parent_path, 'configs', 'deepspeed', zero_stage_file)
        
        import json
        with open(ds_cfgs_path) as f:
            ds_cfgs = json.load(f)
        
        os.environ['ZERO_STAGE'] = str(ds_cfgs['zero_optimization']['stage'])
    
    return cfg, ds_cfgs


# For backward compatibility, create a function that mimics the original read_cfgs
def read_cfgs_with_hydra(mode: str, task: str, overrides: Optional[list] = None) -> tuple[dict, dict]:
    """
    Backward compatible function that returns dict instead of DictConfig.
    """
    cfg, ds_cfgs = read_hydra_cfgs(task, mode=mode, overrides=overrides)
    
    # Convert DictConfig to dict for backward compatibility
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    return cfg_dict, ds_cfgs
