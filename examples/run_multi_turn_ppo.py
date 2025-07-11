#!/usr/bin/env python3
"""
Example script showing how to run multi-turn PPO training with Hydra configuration.
"""

import argparse
import sys
import os

from align_anything.trainers.text_to_text.ppo_multi_turn import PPOTrainer
from align_anything.utils.config_utils import read_hydra_cfgs, ConfigManager
from align_anything.utils.tools import dict_to_namedtuple, seed_everything
from omegaconf import OmegaConf
import deepspeed
from align_anything.utils.device_utils import get_current_device, torch_set_device


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-turn PPO Training with Hydra')
    parser.add_argument('--config-name', type=str, default='ppo_multi_turn',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--model_path', type=str,
                       help='Path to the model (overrides config)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (overrides config)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    args, unknown = parser.parse_known_args()
    
    # Setup distributed training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)
    
    # Build overrides list from command line arguments
    overrides = []
    if args.model_path:
        overrides.append(f"model_path={args.model_path}")
    if args.output_dir:
        overrides.append(f"train_cfgs.output_dir={args.output_dir}")
    
    # Parse additional overrides from unknown args
    for i in range(0, len(unknown), 2):
        if i + 1 < len(unknown):
            key = unknown[i].lstrip('--').replace('-', '_')
            value = unknown[i + 1]
            overrides.append(f"{key}={value}")
    
    # Load configuration using Hydra
    try:
        cfg, ds_cfgs = read_hydra_cfgs(args.config_name, overrides=overrides)
        dict_cfgs = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        print(f"Error loading Hydra config: {e}")
        print("Falling back to legacy config loading...")
        
        # Fallback to legacy config loading
        from align_anything.utils.tools import read_cfgs
        task = os.path.join('text_to_text', args.config_name)
        try:
            dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)
        except:
            # If task-specific config doesn't exist, use general ppo config
            task = os.path.join('text_to_text', 'ppo')
            dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)
            
            # Override with multi-turn settings
            dict_cfgs['train_cfgs']['multi_turn'] = True
            dict_cfgs['train_cfgs']['bi_level_gae'] = True
            dict_cfgs['train_cfgs']['high_level_gamma'] = 0.95
            dict_cfgs['train_cfgs']['max_turn'] = 3
    
    # Parse additional command line arguments
    for i in range(0, len(unknown), 2):
        if i + 1 < len(unknown):
            key = unknown[i].lstrip('--').replace('-', '_')
            value = unknown[i + 1]
            
            # Try to convert to appropriate type
            try:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            
            # Update config
            if 'train_cfgs' not in dict_cfgs:
                dict_cfgs['train_cfgs'] = {}
            dict_cfgs['train_cfgs'][key] = value
    
    # Convert to named tuple
    cfgs = dict_to_namedtuple(dict_cfgs)
    
    # Set random seed
    seed_everything(cfgs.train_cfgs.seed)
    
    print("=" * 60)
    print("Starting Multi-turn PPO Training with RAGEN")
    print("=" * 60)
    print(f"Multi-turn enabled: {getattr(cfgs.train_cfgs, 'multi_turn', False)}")
    print(f"Bi-level GAE: {getattr(cfgs.train_cfgs, 'bi_level_gae', False)}")
    print(f"Max turns: {getattr(cfgs.train_cfgs, 'max_turn', 1)}")
    print(f"High-level gamma: {getattr(cfgs.train_cfgs, 'high_level_gamma', 1.0)}")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = PPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save()
        
        print("=" * 60)
        print("Multi-turn PPO Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
