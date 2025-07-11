# support multi-turn RL, modified from RAGEN

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
"""Trainer for PPO training."""


import argparse
import copy
import itertools
import os
import sys
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text import (
    PromptOnlyBatch,
    PromptOnlyDataset,
    SupervisedDataset,
)
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.base import RLTrainerBase
from align_anything.utils.device_utils import get_current_device, torch_gc, torch_set_device
from align_anything.utils.multi_process import (
    get_all_reduce_max,
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    batch_retokenize,
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    gather_log_probabilities,
    is_same_tokenizer,
    masked_mean,
    prepare_ds_eval_cfgs,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)
from align_anything.utils.config_utils import read_hydra_cfgs, ConfigManager
from omegaconf import OmegaConf
from align_anything.utils.ragen_utils.core_algos import compute_bi_level_gae_advantage_return
from align_anything.utils.ragen_utils.llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg, ApiCallingWrapperWg
from align_anything.utils.ragen_utils.llm_agent.ctx_manager import ContextManager
from align_anything.utils.ragen_utils.llm_agent.es_manager import EnvStateManager
from align_anything.utils.ragen_utils.verl_compat import DataProto


# Try to import from RAGEN utils, with fallbacks
try:
    from align_anything.utils.ragen_utils.llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg, ApiCallingWrapperWg
    RAGEN_AVAILABLE = True
except ImportError:
    RAGEN_AVAILABLE = False
    LLMAgentProxy = None
    VllmWrapperWg = None
    ApiCallingWrapperWg = None


class PPOTrainer(RLTrainerBase):  # pylint: disable=too-many-instance-attributes
    """Trainer base class for PPO training."""

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.ds_eval_cfgs = prepare_ds_eval_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0
        self.use_ptx = False # not elegant
        
        # Multi-turn specific configurations
        self.multi_turn = getattr(self.cfgs.train_cfgs, 'multi_turn', False)
        self.bi_level_gae = getattr(self.cfgs.train_cfgs, 'bi_level_gae', False)
        self.high_level_gamma = getattr(self.cfgs.train_cfgs, 'high_level_gamma', 1.0)
        self.max_turn = getattr(self.cfgs.train_cfgs, 'max_turn', 3)

        self.init_check()
        dist.barrier()
        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}
        self.reward_infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}
        self.init_models()
        if hasattr(self.actor_model, 'infer_batch'):
            self.infer_batch = self.actor_model.infer_batch
        if hasattr(self.reward_model, 'infer_batch'):
            self.reward_infer_batch = self.reward_model.infer_batch
        dist.barrier()
        self.init_logger()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()


        self.kl_coeff = self.cfgs.train_cfgs.kl_coeff
        self.clip_range_ratio = self.cfgs.train_cfgs.clip_range_ratio
        self.clip_range_score = self.cfgs.train_cfgs.clip_range_score
        self.clip_range_value = self.cfgs.train_cfgs.clip_range_value
        self.ptx_coeff = self.cfgs.train_cfgs.ptx_coeff
        self.gamma = self.cfgs.train_cfgs.gamma
        self.gae_lambda = self.cfgs.train_cfgs.gae_lambda
        
        # Initialize agent proxy for multi-turn rollouts if needed
        if self.multi_turn:
            self.init_agent_proxy()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)
        # loading actor model
        self.bnb_cfgs = self.cfgs.bnb_cfgs
        self.lora_cfgs = self.cfgs.lora_cfgs
        self.actor_model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading actor reference model
        self.actor_reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading reward model (optional for multi-turn)
        self.use_reward_model = not self.multi_turn or getattr(self.cfgs.model_cfgs, 'use_reward_model', False)
        if self.use_reward_model:
            self.reward_model, self.reward_tokenizer, _ = load_pretrained_models(
                self.cfgs.model_cfgs.reward_model_name_or_path,
                model_max_length=self.cfgs.model_cfgs.model_max_length,
                padding_side='right',
                trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
                is_reward_model=True,
                processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
            )
        else:
            self.reward_model = None
            self.reward_tokenizer = self.tokenizer
        # loading reward critic model (optional for multi-turn)
        self.use_reward_critic = not self.multi_turn or getattr(self.cfgs.model_cfgs, 'use_reward_critic', False)
        if self.use_reward_critic:
            self.reward_critic_model, self.reward_critic_tokenizer, _ = load_pretrained_models(
                self.cfgs.model_cfgs.reward_critic_model_name_or_path,
                model_max_length=self.cfgs.model_cfgs.model_max_length,
                padding_side='left',
                trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
                is_reward_model=True,
                processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
            )
        else:
            self.reward_critic_model = None
            self.reward_critic_tokenizer = self.tokenizer
        # initial checking
        if self.use_reward_model and is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer
        if self.use_reward_critic and not is_same_tokenizer(self.tokenizer, self.reward_critic_tokenizer):
            raise ValueError(
                (
                    'Reward critic tokenizer must be the same as actor tokenizer. '
                    'Expected {0.__module__}.{0.__qualname__}(vocab_size={1}), '
                    'but got {2.__module__}.{2.__qualname__}(vocab_size={3}). '
                    'Please consider pass `--reward_critic_model_name_or_path` from the command line.'
                ).format(
                    type(self.tokenizer),
                    len(self.tokenizer),
                    type(self.reward_critic_tokenizer),
                    len(self.reward_critic_tokenizer),
                ),
            )

        # training setup
        if self.use_reward_critic:
            self.reward_critic_tokenizer = self.tokenizer
        self.generation_config = GenerationConfig(
            max_length=self.cfgs.model_cfgs.model_max_length,
            temperature=self.cfgs.model_cfgs.temperature,
            top_p=self.cfgs.model_cfgs.top_p,
            repetition_penalty=self.cfgs.model_cfgs.repetition_penalty,
            do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def init_check(self) -> None:
        """Initial configuration checking."""
        super().init_check()
        if (
            self.cfgs.train_cfgs.per_device_prompt_batch_size
            % self.cfgs.train_cfgs.per_device_train_batch_size
            != 0
        ):
            raise ValueError(
                'The number of prompt-only samples must be divisible by the micro batch size.',
            )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        # For multi-turn mode with RAGEN data, use special data loading
        if self.multi_turn and self.is_ragen_data():
            self.init_ragen_datasets()
        else:
            # Standard data loading
            self.prompt_only_dataloader, self.eval_dataloader, self.ptx_dataloader = (
                self.get_dataloaders(PromptOnlyDataset, PromptOnlyDataset, SupervisedDataset)
            )

    def is_ragen_data(self) -> bool:
        """Check if using RAGEN data path."""
        train_datasets = self.cfgs.data_cfgs.train_datasets
        if isinstance(train_datasets, list):
            return any("ragen_utils/data" in dataset for dataset in train_datasets)
        elif isinstance(train_datasets, str):
            return "ragen_utils/data" in train_datasets
        return False

    def init_ragen_datasets(self) -> None:
        """Initialize datasets for RAGEN data format."""
        # Import RAGEN data loader if available
        from align_anything.utils.ragen_utils.data import load_ragen_data
        
        # Load RAGEN format data
        train_data_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "utils", "ragen_utils", "data", "train"
        )
        eval_data_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "utils", "ragen_utils", "data", "val"
        )
        
        # Create simplified dataloaders for RAGEN data
        # In multi-turn mode, the actual data comes from environment interaction
        # These are just placeholder dataloaders for prompts
        
        class RAGENPromptDataset(Dataset):
            def __init__(self, data_path, tokenizer, max_length=2048):
                self.tokenizer = tokenizer
                self.max_length = max_length
                # Load simple prompts for environment initialization
                self.prompts = self.load_prompts(data_path)
            
            def load_prompts(self, data_path):
                """Load prompts from RAGEN data directory."""
                if os.path.exists(data_path):
                    # Try to load from various possible file formats
                    import json
                    prompts = []
                    for file_ext in ['.json', '.jsonl', '.txt']:
                        file_path = os.path.join(data_path, f'prompts{file_ext}')
                        if os.path.exists(file_path):
                            if file_ext == '.json':
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        prompts.extend([item.get('prompt', str(item)) for item in data])
                            elif file_ext == '.jsonl':
                                with open(file_path, 'r') as f:
                                    for line in f:
                                        item = json.loads(line.strip())
                                        prompts.append(item.get('prompt', str(item)))
                            elif file_ext == '.txt':
                                with open(file_path, 'r') as f:
                                    prompts.extend([line.strip() for line in f if line.strip()])
                            break
                    
                    if not prompts:
                        # Fallback: create default prompts for environment interaction
                        prompts = [
                            "Let's solve this step by step.",
                            "What should I do next?",
                            "Please help me with this task.",
                            "I need to complete this objective."
                        ]
                    return prompts
                else:
                    # Default prompts if data path doesn't exist
                    return [
                        "Let's solve this step by step.",
                        "What should I do next?", 
                        "Please help me with this task.",
                        "I need to complete this objective."
                    ]
            
            def __len__(self):
                return len(self.prompts)
            
            def __getitem__(self, idx):
                prompt = self.prompts[idx % len(self.prompts)]
                
                # Tokenize the prompt
                tokenized = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': tokenized['input_ids'].squeeze(0),
                    'attention_mask': tokenized['attention_mask'].squeeze(0),
                }
            
            def get_collator(self):
                """Return a simple collator for batching."""
                def collate_fn(batch):
                    input_ids = torch.stack([item['input_ids'] for item in batch])
                    attention_mask = torch.stack([item['attention_mask'] for item in batch])
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                    }
                return collate_fn
        
        # Create datasets
        train_dataset = RAGENPromptDataset(train_data_path, self.tokenizer)
        eval_dataset = RAGENPromptDataset(eval_data_path, self.tokenizer)
        
        # Create dataloaders
        
        self.prompt_only_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=int(self.cfgs.train_cfgs.per_device_prompt_batch_size),
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=eval_dataset.get_collator(),
            sampler=DistributedSampler(eval_dataset, shuffle=False),
            batch_size=int(self.cfgs.train_cfgs.per_device_train_batch_size),
        ) if self.cfgs.data_cfgs.eval_datasets else None
        
        # PTX dataloader (not used in multi-turn mode typically)
        self.ptx_dataloader = DataLoader(
            train_dataset,  # Reuse train dataset
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=int(self.cfgs.train_cfgs.per_device_train_batch_size),
        )
        
        self.logger.print(f"Loaded RAGEN datasets: train={len(train_dataset)}, eval={len(eval_dataset) if eval_dataset else 0}")
            

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()

    def split_ptx_micro_batches(
        self,
        ptx_batch: dict[str, torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        """Split a batch of PTX samples into micro-batches."""
        micro_batches = []
        micro_batch = self.infer_batch(ptx_batch)
        for batch_idx in range(0, micro_batch['input_ids'].size(0)):
            micro_batch = {}
            for key, value in ptx_batch.items():
                micro_batch[key] = value[batch_idx : batch_idx + 1, :]
            micro_batches.append(micro_batch)
        return micro_batches

    def actor_step(self, mini_prompt_only_batch: PromptOnlyBatch) -> dict[str, Any]:
        if self.multi_turn and self.agent_proxy:
            # Use multi-turn rollout
            return self.multi_turn_actor_step(mini_prompt_only_batch)
        else:
            # Use standard single-turn generation
            return self.single_turn_actor_step(mini_prompt_only_batch)

    def multi_turn_actor_step(self, mini_prompt_only_batch: PromptOnlyBatch) -> dict[str, Any]:
        """Perform multi-turn rollout using agent proxy."""
        if not RAGEN_AVAILABLE or self.agent_proxy is None:
            self.logger.print("WARNING: RAGEN not available or agent proxy not initialized. Falling back to single-turn.")
            return self.single_turn_actor_step(mini_prompt_only_batch)
        
        # Create DataProto from mini_prompt_only_batch
        dataproto = DataProto()
        dataproto.batch = {
            'input_ids': mini_prompt_only_batch['input_ids'],
            'attention_mask': mini_prompt_only_batch['attention_mask'],
        }
        dataproto.non_tensor_batch = {}
        dataproto.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True,
            'validate': False
        }
        
        # Perform multi-turn rollout using the agent proxy
        # This will handle the full environment interaction loop
        rollouts = self.agent_proxy.rollout(dataproto, val=False)
        
        # Convert rollouts back to actor_batch format
        actor_batch = {
            'input_ids': rollouts.batch['input_ids'],
            'attention_mask': rollouts.batch['attention_mask'],
        }
        
        # Add additional fields needed for training
        if 'response_mask' in rollouts.batch:
            actor_batch['response_mask'] = rollouts.batch['response_mask']
        if 'loss_mask' in rollouts.batch:
            actor_batch['loss_mask'] = rollouts.batch['loss_mask']
        if 'token_level_rewards' in rollouts.batch:
            actor_batch['token_level_rewards'] = rollouts.batch['token_level_rewards']
        if 'rm_scores' in rollouts.batch:
            # Convert rm_scores to token_level_rewards if not already present
            if 'token_level_rewards' not in actor_batch:
                actor_batch['token_level_rewards'] = rollouts.batch['rm_scores']
            
        return actor_batch
    
    def single_turn_actor_step(self, mini_prompt_only_batch: PromptOnlyBatch) -> dict[str, Any]:
        """Standard single-turn actor step."""
        infer_batch = self.infer_batch(mini_prompt_only_batch)
        actor_batch = copy.deepcopy(infer_batch)
        sequences = self.actor_model.module.generate(
            **infer_batch,
            generation_config=self.generation_config,
            synced_gpus=True,
            do_sample=True,
        )
        attention_mask = sequences.not_equal(self.tokenizer.pad_token_id)
        actor_batch['input_ids'] = sequences
        actor_batch['attention_mask'] = attention_mask
        return actor_batch

    def reward_model_step(self, actor_batch: PromptOnlyBatch) -> dict[str, Any]:
        reward_batch = copy.deepcopy(actor_batch)
        
        # Handle multi-turn reward computation
        if self.multi_turn and 'token_level_rewards' in actor_batch:
            # Use token-level rewards from environment directly
            reward_batch['reward'] = actor_batch['token_level_rewards'].sum(-1)  # Sum across sequence
            reward_batch['token_level_rewards'] = actor_batch['token_level_rewards']
        elif self.use_reward_model:
            # Standard reward model computation
            if self.reward_tokenizer is not self.tokenizer:
                reward_tokenize_output = batch_retokenize(
                    actor_batch['input_ids'],
                    src_tokenizer=self.tokenizer,
                    dest_tokenizer=self.reward_tokenizer,
                    skip_special_tokens=True,
                    device=self.reward_model.device,
                )
                reward_batch['input_ids'] = reward_tokenize_output['input_ids']
                reward_batch['attention_mask'] = reward_tokenize_output['attention_mask']
            reward_infer_batch = self.reward_infer_batch(reward_batch)
            reward_batch['reward'] = self.reward_model(**reward_infer_batch).end_scores.squeeze(dim=-1)
        else:
            # No reward model available, use zero rewards as placeholder
            batch_size = actor_batch['input_ids'].shape[0]
            device = actor_batch['input_ids'].device
            reward_batch['reward'] = torch.zeros(batch_size, device=device)
        
        # Compute value estimates using critic model (if available)
        if self.use_reward_critic:
            critic_infer_batch = self.reward_infer_batch(actor_batch)
            scores = self.reward_critic_model(**critic_infer_batch).scores
            reward_batch['reward_values'] = scores.squeeze(dim=-1)[:, :-1]
        else:
            # No critic model, use zero values as placeholder
            seq_len = actor_batch['input_ids'].shape[1] - 1
            batch_size = actor_batch['input_ids'].shape[0]
            device = actor_batch['input_ids'].device
            reward_batch['reward_values'] = torch.zeros(batch_size, seq_len, device=device)
        
        # Copy masks for multi-turn training
        if 'response_mask' in actor_batch:
            reward_batch['response_mask'] = actor_batch['response_mask']
        if 'loss_mask' in actor_batch:
            reward_batch['loss_mask'] = actor_batch['loss_mask']

        return reward_batch

    @torch.no_grad()
    def rollout(self, prompt_only_batch: PromptOnlyBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        # freeze the model for rolling out
        self.set_train(mode=False)

        total_batch_size = prompt_only_batch['input_ids'].size(0)
        micro_batch_size = int(self.cfgs.train_cfgs.per_device_train_batch_size)
        micro_inference_batches = []
        micro_training_batches = []
        mini_batch = {}
        for i in range(0, total_batch_size, micro_batch_size):

            mini_batch = {
                key: prompt_only_batch[key][i : i + micro_batch_size] for key in prompt_only_batch
            }

            # actor generation
            actor_batch = self.actor_step(mini_batch)
            # reward model and reward critic model scoring
            reward_batch = self.reward_model_step(actor_batch)
            # calculate the log probabilities
            logits = self.actor_model(**actor_batch).logits
            ref_logits = self.actor_reference_model(**actor_batch).logits
            log_probs = gather_log_probabilities(logits[:, :-1], actor_batch['input_ids'][:, 1:])
            ref_log_probs = gather_log_probabilities(
                ref_logits[:, :-1], actor_batch['input_ids'][:, 1:]
            )

            micro_training_batch = {}
            micro_training_batch['prompt_idx'] = mini_batch['input_ids'].size(-1) - 1
            micro_training_batch['log_probs'] = log_probs
            micro_training_batch['ref_log_probs'] = ref_log_probs
            micro_training_batch['reward'] = reward_batch['reward']
            micro_training_batch['reward_values'] = reward_batch['reward_values']
            
            # Add multi-turn specific fields
            if self.multi_turn:
                if 'token_level_rewards' in reward_batch:
                    micro_training_batch['token_level_rewards'] = reward_batch['token_level_rewards']
                if 'response_mask' in reward_batch:
                    micro_training_batch['response_mask'] = reward_batch['response_mask']
                if 'loss_mask' in reward_batch:
                    micro_training_batch['loss_mask'] = reward_batch['loss_mask']

            mini_batch['input_ids'] = reward_batch['input_ids']
            mini_batch['attention_mask'] = actor_batch['attention_mask']
            # add rollout results to the batches
            micro_inference_batches.append(mini_batch)
            micro_training_batches.append(micro_training_batch)

        # unfreeze the model for training
        self.set_train()

        return micro_inference_batches, micro_training_batches

    def actor_loss_fn(
        self,
        log_probs: torch.Tensor,  # size = (B, L - S)
        old_log_probs: torch.Tensor,  # size = (B, L - S)
        advantages: torch.Tensor,  # size = (B, L - S)
        mask: torch.BoolTensor,  # size = (B, L - S)
    ) -> torch.Tensor:  # size = ()
        # size = (B, L - S)
        ratios = torch.exp(log_probs - old_log_probs)
        surrogate1 = advantages * ratios
        surrogate2 = advantages * torch.clamp(
            ratios,
            1.0 - self.clip_range_ratio,
            1.0 + self.clip_range_ratio,
        )
        surrogate = torch.minimum(surrogate1, surrogate2)
        return -masked_mean(surrogate, mask)  # size = ()

    def rl_step(
        self, inference_batch: dict[str, torch.Tensor], training_batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Perform a single update step with RL loss."""
        old_log_probs = training_batch['log_probs']
        ref_log_probs = training_batch['ref_log_probs']
        reward = training_batch['reward']
        old_reward_values = training_batch['reward_values']
        start = training_batch['prompt_idx']

        input_ids = inference_batch['input_ids']
        attention_mask = inference_batch['attention_mask']

        sequence_mask = attention_mask[:, 1:]
        
        # Use response_mask for multi-turn or fallback to start index
        if self.multi_turn and 'response_mask' in training_batch:
            response_mask = training_batch['response_mask']
            loss_mask = training_batch.get('loss_mask', response_mask)
        else:
            response_mask = sequence_mask[:, start:]
            loss_mask = response_mask

        with torch.no_grad():
            if self.multi_turn and 'token_level_rewards' in training_batch:
                # Use bi-level GAE for multi-turn
                if self.bi_level_gae:
                    reward_advantages, reward_returns = compute_bi_level_gae_advantage_return(
                        token_level_rewards=training_batch['token_level_rewards'],
                        values=old_reward_values,
                        loss_mask=loss_mask,
                        gamma=self.gamma,
                        lam=self.gae_lambda,
                        high_level_gamma=self.high_level_gamma,
                    )
                else:
                    # Standard GAE but with token-level rewards
                    old_rewards = self.add_kl_divergence_regularization(
                        reward,
                        old_log_probs,
                        ref_log_probs,
                        sequence_mask,
                    )
                    reward_advantages, reward_returns = self.get_advantages_and_returns(
                        old_reward_values,
                        old_rewards,
                        loss_mask,
                        0,  # start from beginning for multi-turn
                    )
            else:
                # Standard single-turn logic
                old_rewards = self.add_kl_divergence_regularization(
                    reward,
                    old_log_probs,
                    ref_log_probs,
                    sequence_mask,
                )
                reward_advantages, reward_returns = self.get_advantages_and_returns(
                    old_reward_values,
                    old_rewards,
                    sequence_mask,
                    start,
                )

        logits = self.actor_model(**inference_batch, use_cache=False).logits
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        
        # Apply actor loss on the appropriate mask
        if self.multi_turn:
            actor_loss = self.actor_loss_fn(
                log_probs,
                old_log_probs,
                reward_advantages,
                loss_mask,
            )
        else:
            actor_loss = self.actor_loss_fn(
                log_probs[:, start:],
                old_log_probs[:, start:],
                reward_advantages,
                sequence_mask[:, start:],
            )
        
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        # Update critic model only if available
        if self.use_reward_critic:
            reward_values = self.reward_critic_model(**inference_batch).scores
            reward_values = reward_values.squeeze(dim=-1)[:, :-1]
            
            # Apply critic loss on the appropriate mask
            if self.multi_turn:
                reward_critic_loss = self.critic_loss_fn(
                    reward_values,
                    old_reward_values,
                    reward_returns,
                    loss_mask,
                )
            else:
                reward_critic_loss = self.critic_loss_fn(
                    reward_values[:, start:],
                    old_reward_values[:, start:],
                    reward_returns,
                    sequence_mask[:, start:],
                )
            
            self.reward_critic_model.backward(reward_critic_loss)
            self.reward_critic_model.step()
        else:
            # No critic model, set critic loss to zero
            reward_critic_loss = torch.tensor(0.0, device=actor_loss.device)
            reward_values = old_reward_values  # Use old values for logging

        with torch.no_grad():
            if self.multi_turn:
                mask = loss_mask
                kl_divergence = ((old_log_probs - ref_log_probs) * mask).sum(dim=-1).mean()
                mean_generated_length = mask.sum(dim=-1).float().mean()
                max_generated_length = mask.sum(dim=-1).float().max()
                reward_with_kl_penalty = (old_rewards * mask).sum(dim=-1).mean() if 'old_rewards' in locals() else reward.mean()
            else:
                mask = sequence_mask[:, start:]
                kl_divergence = ((old_log_probs - ref_log_probs)[:, start:] * mask).sum(dim=-1).mean()
                mean_generated_length = mask.sum(dim=-1).float().mean()
                max_generated_length = mask.sum(dim=-1).float().max()
                reward_with_kl_penalty = (old_rewards[:, start:] * mask).sum(dim=-1).mean()

            reward = reward.mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values[:, start:], mask)

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            reward = get_all_reduce_mean(reward)
            reward_with_kl_penalty = get_all_reduce_mean(reward_with_kl_penalty)
            reward_advantage = get_all_reduce_mean(reward_advantage)
            reward_return = get_all_reduce_mean(reward_return)
            reward_value = get_all_reduce_mean(reward_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        dist.barrier()

        return {
            'train/actor_loss': actor_loss.item(),
            'train/reward_critic_loss': reward_critic_loss.item(),
            'train/reward': reward.item(),
            'train/reward_with_kl_penalty': reward_with_kl_penalty.item(),
            'train/reward_advantage': reward_advantage.item(),
            'train/reward_return': reward_return.item(),
            'train/reward_value': reward_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
        }

    def ptx_step(self, ptx_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Perform a single update step with PTX loss."""
        ptx_loss = self.actor_model(**self.infer_batch(ptx_batch)).loss
        self.actor_model.backward(self.ptx_coeff * ptx_loss)
        self.actor_model.step()
        ptx_loss = get_all_reduce_mean(ptx_loss)
        return {
            'train/ptx_loss': ptx_loss.item(),
        }

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.total_training_steps,
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.eval()

        num_prompt_only_batches = len(self.prompt_only_dataloader)
        num_ptx_batches = len(self.ptx_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches
        for epoch in range(int(self.cfgs.train_cfgs.epochs)):
            for prompt_only_batch, ptx_batch in zip(
                self.prompt_only_dataloader,
                itertools.chain.from_iterable([self.ptx_dataloader] * num_ptx_replicas),
            ):
                inference_batches, training_batches = self.rollout(prompt_only_batch)

                if self.use_ptx:
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                else:
                    ptx_batches = [None for _ in range(len(inference_batches))]
                torch_gc()

                for _ in range(self.cfgs.train_cfgs.update_iters):
                    for inference_batch, training_batch, ptx_batch in zip(
                        inference_batches, training_batches, ptx_batches
                    ):
                        rl_info = self.rl_step(inference_batch, training_batch)

                        torch_gc()
                        self.logger.log(rl_info, step=self.global_step)
                        if self.use_ptx:
                            ptx_info = self.ptx_step(ptx_batch)
                            torch_gc()
                            self.logger.log(ptx_info, step=self.global_step)

                        self.global_step += 1
                        progress_bar.set_description(
                            f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                            f'(reward {rl_info["train/reward"]:.4f})',
                        )
                        progress_bar.update(1)

                        save_interval = (
                            self.total_update_steps // self.cfgs.logger_cfgs.save_total_limit
                        )

                        if self.global_step % save_interval == 0:
                            self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                            self.save(tag=self.global_step)
                            self.logger.print('Checkpoint saved.')

                        if (
                            self.cfgs.data_cfgs.eval_datasets
                            and self.cfgs.train_cfgs.eval_strategy == 'steps'
                            and self.global_step % self.cfgs.train_cfgs.eval_interval == 0
                        ):
                            self.logger.print(
                                f'\n***** Evaluating at step {self.global_step} *****',
                            )
                            self.eval()

            if self.cfgs.data_cfgs.eval_datasets and self.cfgs.train_cfgs.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                self.eval()

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        sequence_mask: torch.BoolTensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.size(-1)
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def critic_loss_fn(
        self,
        values: torch.Tensor,  # size = (B, L - S)
        old_values: torch.Tensor,  # size = (B, L - S)
        returns: torch.Tensor,  # size = (B, L - S)
        mask: torch.BoolTensor,  # size = (B, L - S)
    ) -> torch.Tensor:  # size = ()
        """Compute critic loss."""
        # size = (B, L - S)
        values_clipped = torch.clamp(
            values,
            old_values - self.clip_range_value,
            old_values + self.clip_range_value,
        )
        vf_loss1 = torch.square(values - returns)
        vf_loss2 = torch.square(values_clipped - returns)
        return 0.5 * masked_mean(torch.maximum(vf_loss1, vf_loss2), mask)  # size = ()

    def add_kl_divergence_regularization(
        self,
        reward: torch.Tensor,  # size = (B,)
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
    ) -> torch.Tensor:  # size = (B, L)
        """Add KL divergence regularization on scalar rewards."""
        end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # size = (B,)

        # size = (B, L)
        kl_divergence_estimate = log_probs - ref_log_probs
        kl_penalty_rewards = -self.kl_coeff * kl_divergence_estimate
        rewards = torch.scatter_add(
            kl_penalty_rewards,
            dim=-1,
            index=end_index.unsqueeze(dim=-1),
            src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        )
        return torch.clamp(rewards, min=-self.clip_range_score, max=self.clip_range_score)

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.save_transformers(model=model, tag=tag)

    def init_agent_proxy(self) -> None:
        """Initialize agent proxy for multi-turn rollouts."""
        if not RAGEN_AVAILABLE:
            self.logger.print("WARNING: RAGEN utils not available. Multi-turn functionality will be limited.")
            self.agent_proxy = None
            return
            
        # Initialize context and environment state managers
        self.train_ctx_manager = ContextManager(self.cfgs, self.tokenizer, mode="train")
        self.train_es_manager = EnvStateManager(self.cfgs, mode="train") 
        self.val_ctx_manager = ContextManager(self.cfgs, self.tokenizer, mode="val")
        self.val_es_manager = EnvStateManager(self.cfgs, mode="val")
        
        # Create a simple wrapper for the actor model to work with agent proxy
        class SimpleActorWrapper:
            def __init__(self, actor_model, tokenizer, generation_config):
                self.actor_model = actor_model
                self.tokenizer = tokenizer
                self.generation_config = generation_config
            
            def generate_sequences(self, lm_inputs):
                """Generate sequences using the actor model."""
                input_ids = lm_inputs.batch['input_ids']
                attention_mask = lm_inputs.batch['attention_mask']
                
                # Generate using the actor model
                with torch.no_grad():
                    sequences = self.actor_model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=self.generation_config,
                        synced_gpus=True,
                        do_sample=True,
                    )
                
                # Decode the generated sequences to text
                response_texts = []
                for i, seq in enumerate(sequences):
                    # Extract only the generated part (after the input)
                    input_len = input_ids[i].shape[0]
                    generated_seq = seq[input_len:]
                    response_text = self.tokenizer.decode(generated_seq, skip_special_tokens=True)
                    response_texts.append(response_text)
                
                # Create output DataProto
                lm_outputs = DataProto()
                lm_outputs.non_tensor_batch = {
                    'response_texts': response_texts,
                    'env_ids': lm_inputs.non_tensor_batch.get('env_ids', list(range(len(response_texts)))),
                    'group_ids': lm_inputs.non_tensor_batch.get('group_ids', list(range(len(response_texts))))
                }
                lm_outputs.meta_info = lm_inputs.meta_info
                return lm_outputs
        
        actor_wg = SimpleActorWrapper(self.actor_model, self.tokenizer, self.generation_config)
        
        # Create agent proxy with environment managers
        self.agent_proxy = LLMAgentProxy(
            config=self.cfgs,
            actor_rollout_wg=actor_wg,
            tokenizer=self.tokenizer
        )
        
        # Override the agent proxy's managers with our own
        self.agent_proxy.train_ctx_manager = self.train_ctx_manager
        self.agent_proxy.train_es_manager = self.train_es_manager
        self.agent_proxy.val_ctx_manager = self.val_ctx_manager
        self.agent_proxy.val_es_manager = self.val_es_manager
            


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-turn PPO Training with Hydra')
    parser.add_argument('--config-name', type=str, default='ppo_multi_turn',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    args, unparsed_args = parser.parse_known_args()
    
    # Setup distributed training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)
    
    # Load configuration using Hydra
    cfg, ds_cfgs = read_hydra_cfgs(args.config_name, task="multi_turn", overrides=None)
    dict_cfgs = OmegaConf.to_container(cfg, resolve=True)
    
    # Parse additional command line arguments
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))
    
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
    
    # Initialize trainer
    trainer = PPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
