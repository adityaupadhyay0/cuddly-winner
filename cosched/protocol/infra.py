from typing import Optional
from pydantic import BaseModel, Field, model_validator
from cosched.protocol.distribution import Distribution
from cosched.protocol.relations import ResourceType, PressureVector, Locality, ConstraintSet


class InfraState(BaseModel):
    """
    Time-bounded observation/estimate of infrastructure state (§4.5).
    """
    resource: ResourceType
    capacity: Distribution
    availability: Distribution
    expected_wait: Distribution
    pressure: PressureVector = Field(default_factory=PressureVector)
    locality: Optional[Locality] = None
    constraints: ConstraintSet = Field(default_factory=ConstraintSet)
    issued_at: float = Field(description="Timestamp in seconds when state was issued")
    valid_until: float = Field(description="Timestamp in seconds when state expires")

    @model_validator(mode="after")
    def check_validity_window(self) -> "InfraState":
        if self.valid_until < self.issued_at:
            raise ValueError(f"valid_until ({self.valid_until}) cannot be earlier than issued_at ({self.issued_at})")
        return self
