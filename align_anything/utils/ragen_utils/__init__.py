"""
RAGEN utilities for align-anything multi-turn training.

This module provides compatibility functions and classes for integrating
RAGEN's multi-turn capabilities into align-anything.
"""

# Export verl compatibility functions
from .verl_compat import (
    DataProto,
    masked_whiten,
    compute_response_mask,
    apply_kl_penalty,
    pad_dataproto_to_divisor,
    unpad_dataproto,
    collate_fn,
    compute_log_probs,
    compute_advantages,
    get_reward_model_scores,
    create_env_inputs,
    create_multi_turn_batch,
    split_prompt_response,
    create_rollout_batch,
    load_model_for_inference,
    format_conversation_for_model
)

# Try to import core components
try:
    from .core_algos import compute_bi_level_gae_advantage_return
    HAS_CORE_ALGOS = True
except ImportError:
    HAS_CORE_ALGOS = False

try:
    from .llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg, ApiCallingWrapperWg
    HAS_AGENT_PROXY = True
except ImportError:
    HAS_AGENT_PROXY = False

__all__ = [
    'DataProto',
    'masked_whiten', 
    'compute_response_mask',
]

if HAS_CORE_ALGOS:
    __all__.append('compute_bi_level_gae_advantage_return')

if HAS_AGENT_PROXY:
    __all__.extend(['LLMAgentProxy', 'VllmWrapperWg', 'ApiCallingWrapperWg'])
