from dataclasses import dataclass
from typing import Optional


@dataclass
class KVRetentionDecision:
    workflow_id: str
    retain: bool
    expected_recompute_cost: float
    expected_opportunity_cost: float
    effective_probability: float
    expected_hold_time: float


def decide_kv_retention(
    workflow_id: str,
    kv_size_gb: float,
    expected_hold_time: float,
    effective_probability: float,
    gpu_memory_opportunity_cost_per_gb_sec: float,
    recompute_or_reload_cost: float,
) -> KVRetentionDecision:
    """
    Computes KV retain/evict decision according to CoSched v0.3 SRS §7.2:

    RetainCost ≈ memory_opportunity_cost * KV_size * expected_hold_time
    EvictCost  ≈ recompute_or_reload_cost * probability_of_return
    Retain if  EvictCost > RetainCost
    """
    retain_cost = gpu_memory_opportunity_cost_per_gb_sec * kv_size_gb * expected_hold_time
    evict_cost = recompute_or_reload_cost * effective_probability

    retain = evict_cost > retain_cost

    return KVRetentionDecision(
        workflow_id=workflow_id,
        retain=retain,
        expected_recompute_cost=evict_cost,
        expected_opportunity_cost=retain_cost,
        effective_probability=effective_probability,
        expected_hold_time=expected_hold_time,
    )
