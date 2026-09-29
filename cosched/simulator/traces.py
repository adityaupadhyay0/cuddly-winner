from dataclasses import dataclass, field
from typing import List, Optional, Dict
from uuid import UUID, uuid4
from cosched.protocol.relations import RelationSet


@dataclass
class StepEvent:
    step_id: str
    step_type: str  # "INFERENCE", "TOOL_WAIT", "CPU_WORK", "BRANCH", "DONE"
    duration: float  # Expected duration in seconds
    gpu_tokens: float = 0.0  # Quantity of GPU work if INFERENCE
    kv_size_gb: float = 1.0  # KV memory footprint
    recompute_cost: float = 2.0  # Cost in seconds or monetary units if KV evicted
    probability: float = 1.0  # Probability of executing this step/branch
    branch_options: List["StepEvent"] = field(default_factory=list)
    relation: Optional[RelationSet] = None


@dataclass
class AgentWorkloadTrace:
    workflow_id: UUID = field(default_factory=uuid4)
    tenant_id: UUID = field(default_factory=uuid4)
    agent_class: str = "default_agent"
    steps: List[StepEvent] = field(default_factory=list)
    coordination_group: Optional[str] = None
    deadline: Optional[float] = None
