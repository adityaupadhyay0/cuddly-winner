from typing import Dict, Optional, Tuple, Set
from uuid import UUID
from cosched.protocol.demand import DemandForecast
from cosched.protocol.infra import InfraState


class LifecycleManager:
    """
    Manages state, revision ordering, and expiry for DemandForecasts and InfraState snapshots (§5.3 - §5.5).
    """

    def __init__(self):
        # Maps forecast_id (UUID) -> latest valid DemandForecast
        self._active_forecasts: Dict[UUID, DemandForecast] = {}
        # Count late / superseded / invalid revisions for telemetry
        self.ignored_late_revisions_count: int = 0
        self.superseded_count: int = 0
        # Maps ResourceType -> latest InfraState
        self._active_infra_states: Dict[str, InfraState] = {}

    def ingest_forecast(self, forecast: DemandForecast, current_time: float) -> Tuple[bool, str]:
        """
        Ingests a forecast revision.
        Enforces revision ordering (strictly increasing per ID) and expiry checks.
        Returns (accepted: bool, reason: str).
        """
        # Check if already expired at current_time
        if forecast.valid_until <= current_time:
            return False, "expired_on_arrival"

        existing = self._active_forecasts.get(forecast.id)
        if existing is not None:
            if forecast.revision <= existing.revision:
                self.ignored_late_revisions_count += 1
                return False, f"late_or_duplicate_revision (got {forecast.revision}, current {existing.revision})"

        # Check if explicitly invalidated via probability 0 (§5.3)
        if forecast.probability == 0.0:
            if forecast.id in self._active_forecasts:
                del self._active_forecasts[forecast.id]
            return True, "explicitly_invalidated"

        if existing is not None:
            self.superseded_count += 1

        self._active_forecasts[forecast.id] = forecast
        return True, "accepted"

    def get_active_forecasts(self, current_time: float) -> Dict[UUID, DemandForecast]:
        """
        Returns all active forecasts that have not expired. Purges expired forecasts.
        """
        expired_ids: Set[UUID] = set()
        active: Dict[UUID, DemandForecast] = {}

        for fid, forecast in self._active_forecasts.items():
            if forecast.valid_until <= current_time:
                expired_ids.add(fid)
            else:
                active[fid] = forecast

        for fid in expired_ids:
            del self._active_forecasts[fid]

        return active

    def ingest_infra_state(self, state: InfraState, current_time: float) -> Tuple[bool, str]:
        """
        Ingests infrastructure state update.
        """
        if state.valid_until <= current_time:
            return False, "expired_on_arrival"

        key = state.resource.value
        existing = self._active_infra_states.get(key)
        if existing is not None and state.issued_at <= existing.issued_at:
            return False, "stale_or_duplicate_infra_state"

        self._active_infra_states[key] = state
        return True, "accepted"

    def get_active_infra_state(self, resource_type: str, current_time: float) -> Optional[InfraState]:
        state = self._active_infra_states.get(resource_type)
        if state is None:
            return None
        if state.valid_until <= current_time:
            del self._active_infra_states[resource_type]
            return None
        return state
