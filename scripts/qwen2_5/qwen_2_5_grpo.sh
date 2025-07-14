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


ACTOR_MODEL_NAME_OR_PATH="/home/adminad/kangshijia/models/Qwen2.5-1.5B-Instruct" # actor model path
# REWARD_MODEL_NAME_OR_PATH="/home/adminad/hy/align-anything/outputs/qwen_2_5_rm/slice_end" # reward model path

TRAIN_DATASETS="/home/adminad/hy/dataset/processed/math/train.json" # dataset path
TRAIN_TEMPLATE="PKUSafeRLHF" # rlhf dataset template
TRAIN_SPLIT="train" # split the rlhf dataset

OUTPUT_DIR="../output/llama_grpo" # output dir
# For wandb online logging
export WANDB_API_KEY="8354c6c9b0295dcdc0e86332397f382e265fbd8c"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.trainers.text_to_text.grpo \
  --enable_reward_model False \
  --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
  --train_datasets ${TRAIN_DATASETS} \
  --train_split ${TRAIN_SPLIT} \
  --train_template ${TRAIN_TEMPLATE} \
  --output_dir ${OUTPUT_DIR}
