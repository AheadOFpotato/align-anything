"""
Multi-turn PPO training configuration example.

This configuration file shows how to enable multi-turn RL training 
with environment integration and bi-level GAE.
"""

# Example configuration additions for multi-turn PPO
MULTI_TURN_CONFIG = {
    "train_cfgs": {
        # Enable multi-turn training
        "multi_turn": True,
        
        # Use bi-level GAE for hierarchical advantage estimation
        "bi_level_gae": True,
        
        # High-level gamma for turn-level discounting
        "high_level_gamma": 0.95,
        
        # Maximum number of turns in multi-turn episodes
        "max_turn": 3,
        
        # Standard PPO parameters
        "gamma": 1.0,  # Token-level discounting
        "gae_lambda": 1.0,
        "kl_coeff": 0.02,
        
        # Training parameters
        "per_device_train_batch_size": 4,
        "per_device_prompt_batch_size": 8,
        "update_iters": 1,
        "epochs": 1,
    },
    
    # Environment configuration (if using RAGEN environments)
    "env_cfgs": {
        "env_name": "frozen_lake",  # or other supported environments
        "max_episode_length": 50,
        "reward_shaping": True,
    },
    
    # Agent proxy configuration
    "agent_proxy": {
        "max_turn": 3,
        "use_environment": True,
    },
    
    # Model configurations
    "model_cfgs": {
        "actor_model_name_or_path": "path/to/your/model",
        "reward_model_name_or_path": "path/to/your/reward/model", 
        "reward_critic_model_name_or_path": "path/to/your/critic/model",
        "model_max_length": 2048,
        "temperature": 1.0,
        "top_p": 1.0,
    }
}

"""
Usage:
1. Set multi_turn=True in your config to enable multi-turn training
2. Set bi_level_gae=True to use hierarchical advantage estimation
3. Configure your environment and agent proxy settings
4. Run training with: python -m align_anything.trainers.text_to_text.ppo_multi_turn
"""
