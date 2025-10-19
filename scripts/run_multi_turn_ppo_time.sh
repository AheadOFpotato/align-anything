#!/bin/bash

# Multi-turn PPO training example script
# This script shows how to run the multi-turn PPO trainer

echo "Running multi-turn PPO training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="${PYTHONPATH}:/home/adminad/hy/align-anything"
ACTOR_MODEL_NAME_OR_PATH="/data/huyi/models/Qwen2.5-0.5B-Instruct" # actor model path
CRITIC_MODEL_NAME_OR_PATH="/data/huyi/models/Qwen2.5-0.5B-Instruct" # critic model path

# Training command with multi-turn specific parameters
python -m align_anything.trainers.text_to_text.ppo_multi_turn_time \
    --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
    --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
    --model_cfgs.actor_model_name_or_path "/data/huyi/models/Qwen2.5-0.5B-Instruct" \
    --model_cfgs.reward_model_name_or_path "/data/huyi/models/Qwen2.5-0.5B-Instruct" \
    --model_cfgs.reward_critic_model_name_or_path "/data/huyi/models/Qwen2.5-0.5B-Instruct" \
    --train_cfgs.multi_turn true \
    --train_cfgs.bi_level_gae true \
    --train_cfgs.high_level_gamma 0.95 \
    --train_cfgs.max_turn 3 \
    --train_cfgs.gamma 1.0 \
    --train_cfgs.gae_lambda 1.0 \
    --train_cfgs.kl_coeff 0.02 \
    --train_cfgs.per_device_train_batch_size 4 \
    --train_cfgs.per_device_prompt_batch_size 8 \
    --train_cfgs.update_iters 1 \
    --train_cfgs.epochs 1 \
    --data_cfgs.train_datasets "/home/kangshijia/huawei/align-anything/align_anything/utils/ragen_utils/data" \
    --data_cfgs.eval_datasets "/home/kangshijia/huawei/align-anything/align_anything/utils/ragen_utils/data" \
    --logger_cfgs.save_total_limit 3 \
    --logger_cfgs.save_interval 100

echo "Training completed!"

# Notes:
# - Replace model paths with your actual model paths
# - Replace dataset paths with your actual dataset paths
# - Adjust batch sizes based on your GPU memory
# - Set appropriate values for gamma and lambda based on your environment
# - For RAGEN environments, make sure the environment utils are properly set up
