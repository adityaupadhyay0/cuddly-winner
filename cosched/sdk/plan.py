from uuid import UUID
from typing import List, Optional
from cosched.protocol.demand import DemandForecast
from cosched.protocol.relations import RelationSet
from cosched.sdk.ctx import ctx


class Plan:
    """
    Optional Plan API for managing execution frontiers and relationships (§12.2).
    """

    def __init__(self, workflow_id: UUID, agent_class: str, tenant_id: UUID):
        self.workflow_id = workflow_id
        self.agent_class = agent_class
        self.tenant_id = tenant_id
        self.forecasts: List[DemandForecast] = []

    def add_forecast(self, **kwargs) -> DemandForecast:
        kwargs["workflow_id"] = self.workflow_id
        kwargs["agent_class"] = self.agent_class
        kwargs["tenant_id"] = self.tenant_id
        f = ctx.forecast(**kwargs)
        self.forecasts.append(f)
        return f

    def relate(self, forecast_a: DemandForecast, forecast_b: DemandForecast, relation_type: str = "depends_on"):
        if forecast_a.relation is None:
            forecast_a.relation = RelationSet()
        if relation_type == "depends_on":
            forecast_a.relation.depends_on.append(str(forecast_b.id))
        elif relation_type == "parallel_with":
            forecast_a.relation.parallel_with.append(str(forecast_b.id))
        elif relation_type == "alternative_to":
            forecast_a.relation.alternative_to.append(str(forecast_b.id))
