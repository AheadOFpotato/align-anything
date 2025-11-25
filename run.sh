#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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

conda activate align-anything
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export SWANLAB_API_KEY="9qEe0WqjoMiNBUSLvlT1J"
export task='bandit'
export think='think'
export ASCEND_RT_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="0"
export lr=1e-5



export CUDA_HOME=$CONDA_PREFIX
ACTOR_MODEL_NAME_OR_PATH="~/kangshijia/models/Qwen2.5-0.5B-Instruct" # actor model path
CRITIC_MODEL_NAME_OR_PATH="~/kangshijia/models/Qwen2.5-0.5B-Instruct" # critic model path

TRAIN_DATASETS="./align_anything/utils/ragen_utils/data" # dataset path
TRAIN_SPLIT="train" # split the rlhf dataset


# Source the setup script
source ./scripts/setup.sh

# Execute deepspeed command



deepspeed \
  --master_port 12378 \
  --module align_anything.trainers.text_to_text.ppo_multi_turn_time \
  --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
  --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
  --train_datasets ./align_anything/utils/ragen_utils/data \
  --train_split ${TRAIN_SPLIT} \
  --output_dir "./output/${task}_${lr}_${think}" \
  --actor_lr ${lr} \
  --log_run_name "${task}_${lr}_${think}" 