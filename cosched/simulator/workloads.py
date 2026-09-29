import random
from typing import List, Optional
from uuid import uuid4
from cosched.simulator.traces import AgentWorkloadTrace, StepEvent
from cosched.protocol.relations import RelationSet


class WorkloadGenerator:
    """
    Generates synthetic traces for testing agentic workloads (§11.3).
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_trace(
        self,
        num_steps: int = 4,
        agent_class: str = "react_agent",
        tool_latency_mean: float = 4.0,
        gpu_duration_mean: float = 1.5,
        gpu_tokens_mean: float = 4000.0,
        branch_probability: float = 0.8,
        coordination_group: Optional[str] = None,
        deadline_offset: Optional[float] = 30.0,
    ) -> AgentWorkloadTrace:
        wf_id = uuid4()
        tenant_id = uuid4()
        steps: List[StepEvent] = []

        for idx in range(num_steps):
            # Alternating inference and tool wait
            if idx % 2 == 0:
                # Inference step
                gpu_tokens = self.rng.gauss(gpu_tokens_mean, gpu_tokens_mean * 0.2)
                dur = self.rng.gauss(gpu_duration_mean, gpu_duration_mean * 0.2)
                steps.append(
                    StepEvent(
                        step_id=f"step_{idx}_inf",
                        step_type="INFERENCE",
                        duration=max(0.1, dur),
                        gpu_tokens=max(100, gpu_tokens),
                        kv_size_gb=1.5,
                        recompute_cost=1.2,
                        probability=1.0,
                    )
                )
            else:
                # Tool wait gap
                tool_lat = self.rng.gauss(tool_latency_mean, tool_latency_mean * 0.3)
                prob = branch_probability if idx > 1 else 0.95
                rel = RelationSet(coordination_group=coordination_group) if coordination_group else None
                steps.append(
                    StepEvent(
                        step_id=f"step_{idx}_tool",
                        step_type="TOOL_WAIT",
                        duration=max(0.2, tool_lat),
                        gpu_tokens=0.0,
                        kv_size_gb=1.5,
                        recompute_cost=1.2,
                        probability=prob,
                        relation=rel,
                    )
                )

        steps.append(StepEvent(step_id="step_done", step_type="DONE", duration=0.0))

        return AgentWorkloadTrace(
            workflow_id=wf_id,
            tenant_id=tenant_id,
            agent_class=agent_class,
            steps=steps,
            coordination_group=coordination_group,
            deadline=deadline_offset,
        )
