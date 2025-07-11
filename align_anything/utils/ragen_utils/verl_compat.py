"""
Simplified implementations of verl functions for align-anything multi-turn PPO.
This module provides fallback implementations when verl is not available.
"""

import torch
from typing import Dict, Any, Optional, List, Union


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Whiten values using mask.
    
    Args:
        values: Tensor to whiten
        mask: Mask indicating valid positions
        eps: Small value for numerical stability
        
    Returns:
        Whitened values
    """
    if mask.sum() == 0:
        return values
    
    masked_values = values * mask
    mean = masked_values.sum() / mask.sum()
    
    # Compute variance
    centered = (masked_values - mean * mask)
    variance = (centered * centered * mask).sum() / mask.sum()
    std = torch.sqrt(variance + eps)
    
    # Whiten
    whitened = (values - mean) / std
    return whitened * mask


def compute_response_mask(data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute response mask from batch data.
    
    Args:
        data_batch: Batch containing input_ids and attention_mask
        
    Returns:
        Response mask tensor
    """
    if 'response_mask' in data_batch:
        return data_batch['response_mask']
    
    # Fallback: assume everything after prompt is response
    attention_mask = data_batch['attention_mask']
    if 'prompt_length' in data_batch:
        prompt_lengths = data_batch['prompt_length']
        batch_size, seq_len = attention_mask.shape
        response_mask = torch.zeros_like(attention_mask)
        for i, length in enumerate(prompt_lengths):
            response_mask[i, length:] = attention_mask[i, length:]
        return response_mask[:, 1:]  # Remove first token
    else:
        # Assume first half is prompt, second half is response
        seq_len = attention_mask.shape[1]
        mid_point = seq_len // 2
        response_mask = attention_mask.clone()
        response_mask[:, :mid_point] = 0
        return response_mask[:, 1:]  # Remove first token


# Fallback DataProto class
class DataProto:
    """Simple DataProto implementation for compatibility."""
    
    def __init__(self, batch: Optional[Dict] = None, non_tensor_batch: Optional[Dict] = None, meta_info: Optional[Dict] = None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}


def apply_kl_penalty(batch: Dict[str, torch.Tensor], kl_ctrl: float = 0.1, kl_penalty: str = "kl", multi_turn: bool = False) -> tuple:
    """
    Apply KL penalty to rewards.
    
    Args:
        batch: Batch containing log_probs, ref_log_probs, rewards
        kl_ctrl: KL control coefficient  
        kl_penalty: Type of KL penalty
        multi_turn: Whether this is multi-turn training
        
    Returns:
        Updated batch and KL metrics
    """
    if 'log_probs' not in batch or 'ref_log_probs' not in batch:
        return batch, {}
    
    log_probs = batch['log_probs']
    ref_log_probs = batch['ref_log_probs']
    
    # Compute KL divergence
    kl_div = log_probs - ref_log_probs
    
    # Apply penalty
    if kl_penalty == "kl":
        kl_penalty_tensor = -kl_ctrl * kl_div
    elif kl_penalty == "abs":
        kl_penalty_tensor = -kl_ctrl * torch.abs(kl_div)
    else:
        kl_penalty_tensor = torch.zeros_like(kl_div)
    
    # Add to rewards
    if 'rewards' in batch:
        batch['rewards'] = batch['rewards'] + kl_penalty_tensor.sum(-1)
    
    kl_metrics = {
        'kl_div_mean': kl_div.mean().item(),
        'kl_penalty_mean': kl_penalty_tensor.mean().item()
    }
    
    return batch, kl_metrics


def pad_dataproto_to_divisor(data: DataProto, divisor: int) -> tuple:
    """
    Pad DataProto to make batch size divisible by divisor.
    
    Args:
        data: DataProto to pad
        divisor: Target divisor
        
    Returns:
        Padded DataProto and pad size
    """
    if not data.batch:
        return data, 0
    
    # Get current batch size
    batch_size = None
    for key, value in data.batch.items():
        if torch.is_tensor(value):
            batch_size = value.shape[0]
            break
    
    if batch_size is None:
        return data, 0
    
    # Calculate padding needed
    remainder = batch_size % divisor
    if remainder == 0:
        return data, 0
    
    pad_size = divisor - remainder
    
    # Pad tensors
    padded_data = DataProto()
    padded_data.batch = {}
    padded_data.non_tensor_batch = data.non_tensor_batch.copy()
    padded_data.meta_info = data.meta_info.copy()
    
    for key, value in data.batch.items():
        if torch.is_tensor(value):
            # Repeat last element to pad
            pad_tensor = value[-1:].repeat(pad_size, *([1] * (value.dim() - 1)))
            padded_data.batch[key] = torch.cat([value, pad_tensor], dim=0)
        else:
            padded_data.batch[key] = value
    
    return padded_data, pad_size


def unpad_dataproto(data: DataProto, pad_size: int) -> DataProto:
    """
    Remove padding from DataProto.
    
    Args:
        data: Padded DataProto
        pad_size: Number of padded elements to remove
        
    Returns:
        Unpadded DataProto
    """
    if pad_size == 0:
        return data
    
    unpadded_data = DataProto()
    unpadded_data.batch = {}
    unpadded_data.non_tensor_batch = data.non_tensor_batch.copy()
    unpadded_data.meta_info = data.meta_info.copy()
    
    for key, value in data.batch.items():
        if torch.is_tensor(value):
            unpadded_data.batch[key] = value[:-pad_size]
        else:
            unpadded_data.batch[key] = value
    
    return unpadded_data



def collate_fn(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple collate function for batching data.
    
    Args:
        batch_list: List of data dictionaries
        
    Returns:
        Collated batch dictionary
    """
    if not batch_list:
        return {}
    
    # Get all keys from the first item
    keys = batch_list[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch_list]
        
        # Handle different data types
        if torch.is_tensor(values[0]):
            # Stack tensors
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float)):
            # Convert numbers to tensor
            collated[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            # Keep strings as list
            collated[key] = values
        elif isinstance(values[0], list):
            # Concatenate lists
            collated[key] = [item for sublist in values for item in sublist]
        else:
            # Default: keep as list
            collated[key] = values
    
    return collated


def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute log probabilities from logits and labels.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        mask: Optional mask for valid positions [batch_size, seq_len]
        
    Returns:
        Log probabilities [batch_size, seq_len]
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities for the target labels
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Apply mask if provided
    if mask is not None:
        shift_mask = mask[..., 1:].contiguous()
        gathered_log_probs = gathered_log_probs * shift_mask
    
    return gathered_log_probs


def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95, mask: Optional[torch.Tensor] = None) -> tuple:
    """
    Compute advantages using Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Rewards tensor [batch_size, seq_len]
        values: Value estimates [batch_size, seq_len]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        mask: Optional mask for valid positions
        
    Returns:
        Advantages and returns
    """
    batch_size, seq_len = rewards.shape
    
    # Initialize
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # Work backwards through sequence
    next_value = 0
    next_advantage = 0
    
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_non_terminal = 0
            next_value = 0
        else:
            next_non_terminal = 1
            next_value = values[:, t + 1]
        
        # Compute TD error
        delta = rewards[:, t] + gamma * next_value * next_non_terminal - values[:, t]
        
        # Compute advantage
        advantages[:, t] = delta + gamma * gae_lambda * next_non_terminal * next_advantage
        next_advantage = advantages[:, t]
        
        # Compute returns
        returns[:, t] = advantages[:, t] + values[:, t]
    
    # Apply mask if provided
    if mask is not None:
        advantages = advantages * mask
        returns = returns * mask
    
    return advantages, returns


def get_reward_model_scores(texts: List[str], reward_model=None) -> List[float]:
    """
    Get reward model scores for generated texts.
    
    Args:
        texts: List of generated text strings
        reward_model: Optional reward model (placeholder)
        
    Returns:
        List of reward scores
    """
    # Placeholder implementation - return random scores
    import random
    return [random.uniform(-1.0, 1.0) for _ in texts]


# Environment interaction utilities
def create_env_inputs(responses: List[str], env_ids: List[int], group_ids: List[int]) -> List[Dict]:
    """
    Create environment inputs from LLM responses.
    
    Args:
        responses: List of response strings
        env_ids: List of environment IDs
        group_ids: List of group IDs
        
    Returns:
        List of environment input dictionaries
    """
    env_inputs = []
    for response, env_id, group_id in zip(responses, env_ids, group_ids):
        env_inputs.append({
            'response': response,
            'env_id': env_id,
            'group_id': group_id,
            'action': response  # Assume response is the action
        })
    return env_inputs


# Multi-turn specific utilities
def create_multi_turn_batch(conversations: List[List[Dict]], tokenizer, max_length: int = 2048) -> Dict[str, torch.Tensor]:
    """
    Create a batch from multi-turn conversations.
    
    Args:
        conversations: List of conversations, each conversation is a list of message dicts
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        
    Returns:
        Batch dictionary with tokenized data
    """
    input_ids_list = []
    attention_mask_list = []
    
    for conversation in conversations:
        # Convert conversation to text
        text = ""
        for message in conversation:
            role = message.get('role', 'user')
            content = message.get('content', '')
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids_list.append(encoded['input_ids'].squeeze(0))
        attention_mask_list.append(encoded['attention_mask'].squeeze(0))
    
    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def split_prompt_response(input_ids: torch.Tensor, tokenizer, response_start_token: Optional[int] = None) -> tuple:
    """
    Split input_ids into prompt and response parts.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: Tokenizer used for encoding
        response_start_token: Token ID that marks the start of response
        
    Returns:
        Tuple of (prompt_ids, response_ids, prompt_lengths)
    """
    batch_size, seq_len = input_ids.shape
    prompt_lengths = []
    
    if response_start_token is None:
        # Use a heuristic: find assistant tokens
        if hasattr(tokenizer, 'encode'):
            try:
                assistant_token = tokenizer.encode("<|im_start|>assistant")[0]
            except:
                assistant_token = None
        else:
            assistant_token = None
    else:
        assistant_token = response_start_token
    
    for i in range(batch_size):
        if assistant_token is not None:
            # Find the last occurrence of assistant token
            positions = (input_ids[i] == assistant_token).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                prompt_length = positions[-1].item() + 1
            else:
                prompt_length = seq_len // 2  # Fallback
        else:
            prompt_length = seq_len // 2  # Simple fallback
        
        prompt_lengths.append(prompt_length)
    
    # Create prompt and response tensors
    max_prompt_len = max(prompt_lengths)
    max_response_len = seq_len - min(prompt_lengths)
    
    prompt_ids = torch.zeros(batch_size, max_prompt_len, dtype=input_ids.dtype, device=input_ids.device)
    response_ids = torch.zeros(batch_size, max_response_len, dtype=input_ids.dtype, device=input_ids.device)
    
    for i, prompt_len in enumerate(prompt_lengths):
        prompt_ids[i, :prompt_len] = input_ids[i, :prompt_len]
        response_len = seq_len - prompt_len
        if response_len > 0:
            response_ids[i, :response_len] = input_ids[i, prompt_len:]
    
    return prompt_ids, response_ids, torch.tensor(prompt_lengths)


def create_rollout_batch(rollout_data: List[Dict]) -> DataProto:
    """
    Create a rollout batch from rollout data.
    
    Args:
        rollout_data: List of rollout dictionaries
        
    Returns:
        DataProto containing the rollout batch
    """
    if not rollout_data:
        return DataProto()
    
    # Separate tensor and non-tensor data
    tensor_keys = set()
    non_tensor_keys = set()
    
    for data in rollout_data:
        for key, value in data.items():
            if torch.is_tensor(value) or isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[0]):
                tensor_keys.add(key)
            else:
                non_tensor_keys.add(key)
    
    # Create batch
    batch = {}
    non_tensor_batch = {}
    
    for key in tensor_keys:
        values = [data.get(key) for data in rollout_data if key in data]
        if values and torch.is_tensor(values[0]):
            batch[key] = torch.stack(values, dim=0)
        elif values:
            batch[key] = values
    
    for key in non_tensor_keys:
        values = [data.get(key) for data in rollout_data if key in data]
        non_tensor_batch[key] = values
    
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# Model loading utilities
def load_model_for_inference(model_path: str, device: str = "auto"):
    """
    Load a model for inference.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True
        )
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def format_conversation_for_model(conversation: List[Dict], tokenizer) -> str:
    """
    Format a conversation for the model.
    
    Args:
        conversation: List of message dictionaries
        tokenizer: Tokenizer to use
        
    Returns:
        Formatted conversation string
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(conversation, tokenize=False)
        except:
            pass
    
    # Fallback formatting
    formatted = ""
    for message in conversation:
        role = message.get('role', 'user')
        content = message.get('content', '')
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return formatted
