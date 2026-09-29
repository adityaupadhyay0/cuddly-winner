from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator
from cosched.protocol.distribution import Distribution
from cosched.protocol.relations import ResourceType, RelationSet


class UtilityDescriptor(BaseModel):
    slos_seconds: Optional[float] = None
    value_weight: float = Field(default=1.0, ge=0.0)
    penalty_on_miss: float = Field(default=0.0, ge=0.0)


class DemandForecast(BaseModel):
    """
    Temporal Demand Forecast emitted by an agent/workload (§4.2).
    """
    id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    agent_class: str
    tenant_id: UUID
    resource: ResourceType
    quantity: Distribution
    time_to_need: Distribution
    duration: Distribution
    probability: float = Field(ge=0.0, le=1.0, description="Probability that demand materializes [0,1]")
    deadline: Optional[float] = Field(default=None, description="Absolute timestamp deadline")
    utility: Optional[UtilityDescriptor] = None
    relation: Optional[RelationSet] = None
    issued_at: float = Field(description="Timestamp in seconds when issued")
    revision: int = Field(ge=0, description="Monotonically increasing revision number per forecast id")
    valid_until: float = Field(description="Timestamp in seconds when forecast expires")

    @model_validator(mode="after")
    def check_validity_window(self) -> "DemandForecast":
        if self.valid_until < self.issued_at:
            raise ValueError(f"valid_until ({self.valid_until}) cannot be earlier than issued_at ({self.issued_at})")
        return self
