# Multi-turn PPO Training with RAGEN Integration

本文档介绍如何在align-anything中使用multi-turn PPO训练，并集成RAGEN环境。

## 概述

Multi-turn PPO trainer支持：
- 多轮交互式强化学习训练
- RAGEN环境集成
- 双层GAE (Bi-level Generalized Advantage Estimation)
- Token级别的奖励计算
- 环境状态管理

## 核心组件

### 1. es_manager (EnvStateManager)
- **位置**: `align_anything/utils/ragen_utils/llm_agent/es_manager.py`
- **功能**: 管理环境状态，包括环境重置和步进操作
- **关键方法**:
  - `reset()`: 重置所有环境并返回初始状态
  - `step(env_inputs)`: 执行环境步进，接收动作并返回新状态和奖励
  - `get_rollout_states()`: 获取完整的rollout状态

### 2. ctx_manager (ContextManager)  
- **位置**: `align_anything/utils/ragen_utils/llm_agent/ctx_manager.py`
- **功能**: 管理上下文，处理LLM输入输出与环境的转换
- **关键方法**:
  - `get_lm_inputs(env_outputs)`: 将环境输出转换为LLM输入格式
  - `get_env_inputs(lm_outputs)`: 将LLM输出转换为环境动作
  - `formulate_rollouts(env_outputs)`: 构建最终的rollout数据

### 3. env_outputs
- **格式**: `List[Dict]`，每个字典包含：
  - `env_id`: 环境ID
  - `history`: 历史记录列表
  - `group_id`: 环境组ID
  - `tag`: 环境标签
  - `penalty`: 惩罚值

## 配置说明

### Multi-turn配置
```yaml
train_cfgs:
  multi_turn: true              # 启用multi-turn模式
  bi_level_gae: true           # 启用双层GAE
  high_level_gamma: 0.95       # 高层折扣因子
  max_turn: 3                  # 最大轮数
```

### 环境管理器配置
```yaml
es_manager:
  train:
    env_groups: 2              # 环境组数
    group_size: 4              # 每组环境数量
    env_configs:
      tags: ["FrozenLake", "Sokoban"]
      n_groups: [1, 1]
```

## 环境集成

### 支持的环境
- **FrozenLake**: 网格世界导航
- **Sokoban**: 推箱子游戏
- **Countdown**: 简单计数游戏
- **MetaMathQA**: 数学问题求解
- **Webshop**: 购物场景
- **ALFWorld**: 文本冒险游戏

### 添加新环境

1. 继承`BaseEnv`类：
```python
from align_anything.utils.ragen_utils.env.base import BaseEnv

class MyCustomEnv(BaseEnv):
    def reset(self, seed=None, **kwargs):
        # 实现环境重置逻辑
        pass
    
    def step(self, action):
        # 实现环境步进逻辑
        return observation, reward, done, info
```

## 使用方法

dependencies:
- gymnasium
- hydra-core
- gym
- gym_sokoban
- matplotlib

### 基本训练
```bash
cd align-anything
python examples/run_multi_turn_ppo.py \
    --multi_turn true \
    --bi_level_gae true \
    --max_turn 3 \
    --learning_rate 5e-7
```
