from typing import Any, Dict, Optional
from uuid import UUID, uuid4
from cosched.protocol.demand import DemandForecast
from cosched.protocol.distribution import Distribution
from cosched.protocol.relations import ResourceType, RelationSet


class CoSchedContext:
    """
    Non-blocking SDK context for emitting DemandForecast messages (§12.1).
    Target: p99 <= 20 us write time on hot path (NFR-01).
    """

    def __init__(self, daemon_client: Optional[Any] = None):
        self.daemon_client = daemon_client

    def forecast(
        self,
        workflow_id: UUID,
        agent_class: str,
        tenant_id: UUID,
        resource: ResourceType | str,
        probability: float,
        time_to_need: Dict[str, float] | Distribution,
        quantity: Dict[str, float] | Distribution,
        duration: Dict[str, float] | Distribution,
        issued_at: float,
        valid_for: float = 10.0,
        deadline: Optional[float] = None,
        revision: int = 1,
        relation: Optional[RelationSet] = None,
    ) -> DemandForecast:
        if isinstance(resource, str):
            resource = ResourceType(resource)

        if isinstance(time_to_need, dict):
            time_to_need = Distribution(**time_to_need)
        if isinstance(quantity, dict):
            quantity = Distribution(**quantity)
        if isinstance(duration, dict):
            duration = Distribution(**duration)

        forecast = DemandForecast(
            workflow_id=workflow_id,
            agent_class=agent_class,
            tenant_id=tenant_id,
            resource=resource,
            quantity=quantity,
            time_to_need=time_to_need,
            duration=duration,
            probability=probability,
            deadline=deadline,
            relation=relation,
            issued_at=issued_at,
            revision=revision,
            valid_until=issued_at + valid_for,
        )

        if self.daemon_client is not None:
            # Asynchronous non-blocking dispatch
            self.daemon_client.submit_forecast_async(forecast, now=issued_at)

        return forecast


# Singleton context helper
ctx = CoSchedContext()
