# Multi-turn PPO Trainer for Align-Anything

This document describes the multi-turn PPO trainer that integrates RAGEN's multi-turn capabilities into align-anything.

## Overview

The multi-turn PPO trainer extends the standard PPO implementation to support:
- Multi-turn conversational RL training
- Environment-based reward computation
- Bi-level GAE (Generalized Advantage Estimation)
- Token-level and turn-level reward optimization

## Key Features

### 1. Multi-turn Rollouts
- Uses agent proxy for environment interaction
- Supports multiple conversation turns per episode
- Integrates with RAGEN environments

### 2. Bi-level GAE
- High-level: Turn-based advantage estimation
- Low-level: Token-based advantage estimation
- Hierarchical reward structure

### 3. Flexible Masking
- Response mask for identifying model outputs
- Loss mask for training optimization
- Supports both single-turn and multi-turn scenarios

## Configuration

### Essential Parameters

```yaml
train_cfgs:
  multi_turn: true              # Enable multi-turn training
  bi_level_gae: true           # Use bi-level GAE
  high_level_gamma: 0.95       # Turn-level discount factor
  max_turn: 3                  # Maximum turns per episode
  gamma: 1.0                   # Token-level discount factor
  gae_lambda: 1.0              # GAE lambda parameter
```

### Environment Integration

```yaml
agent_proxy:
  max_turn: 3                  # Max turns for agent proxy
  use_environment: true        # Enable environment interaction

env_cfgs:
  env_name: "frozen_lake"      # Environment name
  max_episode_length: 50       # Max episode length
```

## Usage

### 1. Basic Multi-turn Training

```bash
python -m align_anything.trainers.text_to_text.ppo_multi_turn \
    --train_cfgs.multi_turn true \
    --train_cfgs.bi_level_gae true \
    --train_cfgs.max_turn 3
```

### 2. With Environment Integration

```bash
python -m align_anything.trainers.text_to_text.ppo_multi_turn \
    --train_cfgs.multi_turn true \
    --train_cfgs.bi_level_gae true \
    --env_cfgs.env_name "your_environment"
```

## Key Components

### 1. PPOTrainer (Multi-turn)
- Extends base PPOTrainer with multi-turn capabilities
- Handles environment integration
- Manages bi-level advantage computation

### 2. Agent Proxy
- Manages multi-turn rollouts
- Interfaces with environments
- Handles LLM generation and environment interaction

### 3. Core Algorithms
- `compute_bi_level_gae_advantage_return`: Hierarchical advantage estimation
- Supports both token-level and turn-level rewards

## Dependencies

### Required
- align-anything base framework
- PyTorch and transformers

### Optional (for full functionality)
- RAGEN utils (for environments)
- verl (for advanced features)

## Fallback Behavior

When RAGEN utils are not available:
- Falls back to single-turn training
- Uses simplified DataProto implementation
- Maintains compatibility with standard PPO

## Troubleshooting

### Common Issues

1. **RAGEN not available**: The trainer will fall back to single-turn mode
2. **Environment setup**: Ensure RAGEN environments are properly configured
3. **Memory issues**: Reduce batch sizes for multi-turn training

### Configuration Tips

1. Start with small `max_turn` values (2-3)
2. Adjust `high_level_gamma` based on environment characteristics
3. Use smaller batch sizes for multi-turn training
4. Monitor KL divergence carefully

## Examples

See `scripts/run_multi_turn_ppo.sh` for a complete training example.

## Future Enhancements

- Support for more RAGEN environments
- Advanced reward shaping techniques
- Integration with more sophisticated agent architectures
- Support for mixed single/multi-turn training
