import pytest
from cosched.simulator.workloads import WorkloadGenerator
from cosched.simulator.engine import SimulationEngine
from cosched.policies.kv_retention import decide_kv_retention


def test_kv_retention_policy_math():
    # Low opportunity cost, high recompute cost -> should retain
    d1 = decide_kv_retention(
        workflow_id="wf1",
        kv_size_gb=1.0,
        expected_hold_time=2.0,
        effective_probability=0.9,
        gpu_memory_opportunity_cost_per_gb_sec=0.1,
        recompute_or_reload_cost=2.0,
    )
    # RetainCost = 0.1 * 1 * 2 = 0.2
    # EvictCost  = 2.0 * 0.9 = 1.8
    assert d1.retain

    # High opportunity cost (long hold time), low recompute cost -> should evict
    d2 = decide_kv_retention(
        workflow_id="wf2",
        kv_size_gb=2.0,
        expected_hold_time=20.0,
        effective_probability=0.5,
        gpu_memory_opportunity_cost_per_gb_sec=0.5,
        recompute_or_reload_cost=1.0,
    )
    # RetainCost = 0.5 * 2 * 20 = 20.0
    # EvictCost  = 1.0 * 0.5 = 0.5
    assert not d2.retain


def test_simulation_engine_policies():
    gen = WorkloadGenerator(seed=123)
    traces = [gen.generate_trace(num_steps=4) for _ in range(10)]

    # Run P0 baseline
    engine_p0 = SimulationEngine(policy="P0", num_gpus=2)
    m0 = engine_p0.run(traces)
    assert m0.completed_workflows == 10
    assert m0.jct_p50 > 0.0

    # Run P2 temporal forecasts
    engine_p2 = SimulationEngine(policy="P2", num_gpus=2)
    m2 = engine_p2.run(traces)
    assert m2.completed_workflows == 10
    assert m2.total_forecasts_sent > 0

    # Run Oracle
    engine_oracle = SimulationEngine(policy="ORACLE", num_gpus=2)
    m_oracle = engine_oracle.run(traces)
    assert m_oracle.completed_workflows == 10
