from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    GPU = "GPU"
    CPU = "CPU"
    KV_CACHE = "KV_CACHE"
    MEMORY = "MEMORY"
    NETWORK = "NETWORK"


class RelationSet(BaseModel):
    """
    Explicit relationships among agents, workflows, or frontier elements (§4.6).
    """
    depends_on: List[str] = Field(default_factory=list, description="List of forecast/workflow IDs depended upon")
    parallel_with: List[str] = Field(default_factory=list, description="List of forecast/workflow IDs executing in parallel")
    alternative_to: List[str] = Field(default_factory=list, description="List of alternative branch forecast IDs")
    coordination_group: Optional[str] = Field(default=None, description="ID of shared coordination group or swarm")


class PressureVector(BaseModel):
    """
    Normalized scalars [0, 1] representing infrastructure pressure (§4.5).
    """
    gpu: float = Field(default=0.0, ge=0.0, le=1.0)
    kv: float = Field(default=0.0, ge=0.0, le=1.0)
    queue: float = Field(default=0.0, ge=0.0, le=1.0)
    cpu: float = Field(default=0.0, ge=0.0, le=1.0)
    network: float = Field(default=0.0, ge=0.0, le=1.0)


class Locality(BaseModel):
    node_id: Optional[str] = None
    cluster_id: Optional[str] = None
    rack_id: Optional[str] = None


class ConstraintSet(BaseModel):
    max_batch_size: Optional[int] = None
    tenant_quota_remaining: Optional[float] = None
    node_features: List[str] = Field(default_factory=list)
