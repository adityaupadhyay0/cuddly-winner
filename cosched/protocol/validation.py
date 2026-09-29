from cosched.protocol.demand import DemandForecast
from cosched.protocol.infra import InfraState


def validate_forecast(forecast: DemandForecast) -> None:
    """
    Validates protocol constraints for DemandForecast.
    Pydantic handles basic schema and monotonic quantile validation;
    this function enforces high-level protocol invariant checks.
    """
    if forecast.valid_until < forecast.issued_at:
        raise ValueError("valid_until cannot be earlier than issued_at")
    if not (0.0 <= forecast.probability <= 1.0):
        raise ValueError("probability must be in [0.0, 1.0]")


def validate_infra_state(state: InfraState) -> None:
    """
    Validates protocol constraints for InfraState.
    """
    if state.valid_until < state.issued_at:
        raise ValueError("valid_until cannot be earlier than issued_at")
