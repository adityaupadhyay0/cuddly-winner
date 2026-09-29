import time
from typing import Dict, List, Optional
from uuid import UUID
from cosched.daemon.ingest import IngestValidator
from cosched.protocol.demand import DemandForecast
from cosched.protocol.infra import InfraState
from cosched.policies.kv_retention import decide_kv_retention, KVRetentionDecision


class ReferenceDaemon:
    """
    CoSched Reference Daemon (§3.1).
    Ingests forecasts/infra state, maintains frontier store, and executes policy decision ticks.
    Target tick period: 10 ms (NFR-04).
    """

    def __init__(self, tick_period_ms: float = 10.0):
        self.tick_period_ms = tick_period_ms
        self.ingest = IngestValidator()
        self.decision_history: List[dict] = []

    def submit_forecast_async(self, forecast: DemandForecast, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        accepted, _ = self.ingest.process_forecast(forecast, now)
        return accepted

    def publish_infra_state(self, state: InfraState, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        accepted, _ = self.ingest.process_infra_state(state, now)
        return accepted

    def tick(self, now: Optional[float] = None) -> List[KVRetentionDecision]:
        """
        Executes policy decision tick across active workflow frontiers.
        """
        if now is None:
            now = time.time()

        active_forecasts = self.ingest.lifecycle.get_active_forecasts(now)
        gpu_infra = self.ingest.lifecycle.get_active_infra_state("GPU", now)

        opp_cost = 0.5
        if gpu_infra is not None:
            opp_cost = 0.5 * (1.0 + gpu_infra.pressure.kv)

        decisions = []
        for fid, forecast in active_forecasts.items():
            decision = decide_kv_retention(
                workflow_id=str(forecast.workflow_id),
                kv_size_gb=forecast.quantity.q50,
                expected_hold_time=forecast.time_to_need.q50,
                effective_probability=forecast.probability,
                gpu_memory_opportunity_cost_per_gb_sec=opp_cost,
                recompute_or_reload_cost=2.0,
            )
            decisions.append(decision)
            self.decision_history.append({"tick_time": now, "workflow_id": str(forecast.workflow_id), "retain": decision.retain})

        return decisions
