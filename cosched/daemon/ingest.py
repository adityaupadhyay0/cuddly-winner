from typing import List, Tuple
from cosched.protocol.demand import DemandForecast
from cosched.protocol.infra import InfraState
from cosched.protocol.lifecycle import LifecycleManager
from cosched.protocol.validation import validate_forecast, validate_infra_state


class IngestValidator:
    """
    Ingest & Validation component for Reference Daemon (§3.1).
    Validates schemas, rate-limits, and deduplicates revisions.
    """

    def __init__(self):
        self.lifecycle = LifecycleManager()

    def process_forecast(self, forecast: DemandForecast, current_time: float) -> Tuple[bool, str]:
        try:
            validate_forecast(forecast)
        except Exception as e:
            return False, f"invalid_schema: {str(e)}"

        return self.lifecycle.ingest_forecast(forecast, current_time)

    def process_infra_state(self, state: InfraState, current_time: float) -> Tuple[bool, str]:
        try:
            validate_infra_state(state)
        except Exception as e:
            return False, f"invalid_schema: {str(e)}"

        return self.lifecycle.ingest_infra_state(state, current_time)
