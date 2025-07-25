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
"""Trainer base for RL training."""


import copy
import os
from abc import abstractmethod
from datetime import datetime
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import CONFIG_NAME, PreTrainedModel, get_scheduler

from align_anything.configs.template import ChatTemplate
from align_anything.datasets import DummyDataset
from align_anything.utils.logger import Logger
from align_anything.utils.multi_process import is_main_process
from align_anything.utils.tools import get_optimizer_grouped_parameters, namedtuple_to_dict


class RLTrainerBase:

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the RL trainer base."""
        self.cfgs = cfgs
        self.lora_cfgs = cfgs.lora_cfgs
        self.bnb_cfgs = cfgs.bnb_cfgs
        self.use_ptx = False # not elegant
        self.use_reward_critic = getattr(cfgs.train_cfgs, 'use_reward_critic', True) # not elegant

    def init_check(self) -> None:
        """Initial configuration checking."""
        self.lora_enabled = False
        if self.cfgs.lora_cfgs and self.cfgs.lora_cfgs.use_lora:
            self.lora_enabled = True
            self.save_full_model = self.cfgs.lora_cfgs.save_full_model

    def init_logger(self) -> None:
        """Set logger."""
        logger_cfgs = self.cfgs.logger_cfgs
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.logger = Logger(
            log_type=logger_cfgs.log_type,
            log_dir=logger_cfgs.output_dir,
            log_project=logger_cfgs.log_project,
            log_run_name=f'{logger_cfgs.log_run_name}-{self.cfgs.data_cfgs.train_datasets}-{time}',
            config=namedtuple_to_dict(self.cfgs),
        )
        if self.cfgs.train_cfgs.load_checkpoint:
            self.global_step = int(self.cfgs.model_cfgs.model_name_or_path.split('slice_')[-1])

    @abstractmethod
    def init_models(self) -> None:
        """Initialize model and tokenizer."""

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""

    def get_dataloaders(
        self,
        train_data_dtype,
        eval_data_dtype,
        ptx_data_dtype,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader]:
        """Get the dataloaders based on data_dtype."""
        formatter = self.processor if self.processor else self.tokenizer
        custom_formatter = (
            self.actor_model.apply_chat_template
            if hasattr(self.actor_model, 'apply_chat_template')
            else None
        )
        self.train_template = ChatTemplate(
            formatter, self.cfgs.data_cfgs.train_template, custom_formatter
        )
        self.eval_template = None
        train_dataset = train_data_dtype(
            path=self.cfgs.data_cfgs.train_datasets,
            template=self.train_template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            name=self.cfgs.data_cfgs.train_name,
            size=self.cfgs.data_cfgs.train_size,
            split=self.cfgs.data_cfgs.train_split,
            data_files=self.cfgs.data_cfgs.train_data_files,
            optional_args=self.cfgs.data_cfgs.train_optional_args,
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=int(self.cfgs.train_cfgs.per_device_prompt_batch_size),
        )

        # load ptx datasets
        self.use_ptx = self.cfgs.data_cfgs.ptx_datasets is not None
        if self.use_ptx:
            custom_formatter = (
                self.actor_model.apply_chat_template
                if hasattr(self.actor_model, 'apply_chat_template')
                else None
            )
            self.ptx_template = ChatTemplate(
                formatter, self.cfgs.data_cfgs.ptx_template, custom_formatter
            )
            ptx_dataset = ptx_data_dtype(
                path=self.cfgs.data_cfgs.ptx_datasets,
                template=self.ptx_template,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name=self.cfgs.data_cfgs.ptx_name,
                size=self.cfgs.data_cfgs.ptx_size,
                split=self.cfgs.data_cfgs.ptx_split,
                data_files=self.cfgs.data_cfgs.ptx_data_files,
                optional_args=self.cfgs.data_cfgs.ptx_optional_args,
            )
            ptx_dataloader = DataLoader(
                ptx_dataset,
                collate_fn=ptx_dataset.get_collator(),
                sampler=DistributedSampler(ptx_dataset, shuffle=True),
                batch_size=int(self.cfgs.train_cfgs.per_device_train_batch_size),
            )
        else:
            ptx_dataloader = DataLoader(DummyDataset(len(train_dataloader)))

        if self.cfgs.data_cfgs.eval_datasets:
            custom_formatter = (
                self.actor_model.apply_chat_template
                if hasattr(self.actor_model, 'apply_chat_template')
                else None
            )
            self.eval_template = ChatTemplate(
                formatter, self.cfgs.data_cfgs.eval_template, custom_formatter
            )
            eval_dataset = eval_data_dtype(
                path=self.cfgs.data_cfgs.eval_datasets,
                template=self.eval_template,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name=self.cfgs.data_cfgs.eval_name,
                split=self.cfgs.data_cfgs.eval_split,
                size=self.cfgs.data_cfgs.eval_size,
                data_files=self.cfgs.data_cfgs.eval_data_files,
                optional_args=self.cfgs.data_cfgs.eval_optional_args,
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=int(self.cfgs.train_cfgs.per_device_train_batch_size),
            )
            return train_dataloader, eval_dataloader, ptx_dataloader

        return train_dataloader, None, ptx_dataloader

    def _init_train_deepspeed_engine(
        self,
        model: nn.Module,
        weight_decay: float,
        lr: float,
        lr_scheduler_type: str,
        lr_warmup_ratio: float,
        total_training_steps: int,
        ds_cfgs: dict[str, Any],
    ) -> deepspeed.DeepSpeedEngine:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)
        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=lr,
            betas=self.cfgs.train_cfgs.adam_betas,
        )
        lr_scheduler_update_steps = total_training_steps // ds_cfgs['gradient_accumulation_steps']
        num_warmup_steps = int(lr_scheduler_update_steps * lr_warmup_ratio)
        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=lr_scheduler_update_steps,
        )
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_cfgs,
        )
        return engine

    def _init_eval_deepspeed_engine(
        self,
        model: nn.Module,
        ds_cfgs: dict[str, Any],
    ) -> deepspeed.DeepSpeedEngine:
        engine, *_ = deepspeed.initialize(
            model=model,
            config=ds_cfgs,
        )
        return engine

    def init_deepspeed_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.total_training_steps: int = (
            len(self.prompt_only_dataloader)
            * self.cfgs.train_cfgs.epochs
            * self.cfgs.train_cfgs.update_iters
            * self.cfgs.train_cfgs.per_device_train_batch_size
        )
        self.total_update_steps = (
            self.total_training_steps * self.cfgs.train_cfgs.per_device_prompt_batch_size
        )
        # initialize the actor model engines
        actor_ds_cfgs = copy.deepcopy(self.ds_train_cfgs)
        actor_total_training_steps = self.total_update_steps
        if self.use_ptx:
            actor_ds_cfgs['train_batch_size'] *= 2
            actor_ds_cfgs['gradient_accumulation_steps'] *= 2
            actor_total_training_steps *= 2
        self.actor_model = self._init_train_deepspeed_engine(
            model=self.actor_model,
            weight_decay=self.cfgs.train_cfgs.actor_weight_decay,
            lr=self.cfgs.train_cfgs.actor_lr,
            lr_scheduler_type=self.cfgs.train_cfgs.actor_lr_scheduler_type,
            lr_warmup_ratio=self.cfgs.train_cfgs.actor_lr_warmup_ratio,
            total_training_steps=actor_total_training_steps,
            ds_cfgs=actor_ds_cfgs,
        )
        # initialize the actor reference model engines
        self.actor_reference_model = self._init_eval_deepspeed_engine(
            model=self.actor_reference_model,
            ds_cfgs=self.ds_eval_cfgs,
        )
        self.actor_reference_model.eval()
        # initialize the critic model engines
        if self.use_reward_critic:
            self.reward_critic_model = self._init_train_deepspeed_engine(
                model=self.reward_critic_model,
                weight_decay=self.cfgs.train_cfgs.critic_weight_decay,
                lr=self.cfgs.train_cfgs.critic_lr,
                lr_scheduler_type=self.cfgs.train_cfgs.critic_lr_scheduler_type,
                lr_warmup_ratio=self.cfgs.train_cfgs.critic_lr_warmup_ratio,
                total_training_steps=self.total_update_steps,
                ds_cfgs=self.ds_train_cfgs,
            )
        if (
            self.reward_model is not None
        ):  # NOTE when self.reward_model is None, it means using remote reward model
            self.reward_model = self._init_eval_deepspeed_engine(
                model=self.reward_model,
                ds_cfgs=self.ds_eval_cfgs,
            )
            self.reward_model.eval()
        # setup the gradient checkpointing
        if self.cfgs.train_cfgs.actor_gradient_checkpointing and not self.lora_enabled:
            self.actor_model.gradient_checkpointing_enable()
        if self.cfgs.train_cfgs.critic_gradient_checkpointing and not self.lora_enabled and self.use_reward_critic:
            self.reward_critic_model.gradient_checkpointing_enable()

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            self.actor_model.train()
            if self.use_reward_critic:
                self.reward_critic_model.train()
            if self.cfgs.train_cfgs.actor_gradient_checkpointing and not self.lora_enabled:
                self.actor_model.gradient_checkpointing_enable()
        else:
            self.actor_model.eval()
            if self.use_reward_critic:
                self.reward_critic_model.eval()
            if self.cfgs.train_cfgs.actor_gradient_checkpointing and not self.lora_enabled:
                self.actor_model.gradient_checkpointing_disable()
        return

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

        self.set_train(mode=False)
        prompts: list[str] = []
        generateds: list[str] = []
        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        )
        for batch in eval_dataloader:
            with torch.no_grad():
                seq = self.actor_model.module.generate(
                    **batch,
                    max_length=self.cfgs.model_cfgs.model_max_length,
                    synced_gpus=True,
                    do_sample=True,
                )

            dist.barrier()
            prompt = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            generated = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
            generated = [text[len(prompt[i]) :] for i, text in enumerate(generated)]
            prompts.extend(prompt)
            generateds.extend(generated)
        # Display result in main process
        if is_main_process():
            columns = ['Prompt', 'Generated']
            rows = list(zip(prompts, generateds))
            self.logger.print_table(
                title='Evaluating...',
                columns=columns,
                rows=rows,
                max_num_rows=5,
            )
        dist.barrier()

        self.set_train()

    def save_transformers(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save transformers model and tokenizer in Hugging Face format."""
        dist.barrier()

        if model is None:
            model = self.actor_model  # pylint: disable=no-member

        output_dir = os.path.join(self.cfgs.logger_cfgs.output_dir, f'slice_{tag or "end"}')
        os.makedirs(output_dir, exist_ok=True)

        self.logger.print(f'Saving model to "{output_dir}" ...')

        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        model_to_save: PreTrainedModel = getattr(model, 'module', model)

        if is_main_process():
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_pretrained(output_dir)
            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

        if not self.lora_enabled:
            self.logger.print('Saving 16-bit model...')
            zero_stage = self.ds_train_cfgs.get('zero_optimization', {}).get('stage', 0)
            if zero_stage >= 2:
                save_file_name = 'pytorch_model.bin'
                model.save_16bit_model(output_dir, save_filename=save_file_name)
                if self.cfgs.train_cfgs.save_checkpoint:
                    model.save_checkpoint(output_dir)
            else:
                if is_main_process():
                    model_to_save.save_pretrained(output_dir, is_main_process=True)
        if self.lora_enabled and not self.lora_cfgs.save_full_model:
            self.logger.print('LoRA used. Saving model as LoRA adapters...')
            model.save_pretrained(output_dir)
        if self.lora_enabled and self.lora_cfgs.save_full_model:
            self.logger.print('LoRA used. Saving full model...')
            model = model.module
            model_to_be_saved = model.merge_and_unload()
            model_to_be_saved.save_pretrained(output_dir)

        self.logger.print('Model saved!')
