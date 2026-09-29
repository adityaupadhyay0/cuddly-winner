from typing import Dict, List, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass, field
from cosched.protocol.demand import DemandForecast


@dataclass
class ForecastOutcomePair:
    forecast_id: UUID
    workflow_id: UUID
    tenant_id: UUID
    agent_class: str
    predicted_probability: float
    predicted_time_to_need_q50: float
    predicted_quantity_q50: float
    predicted_duration_q50: float
    actual_materialized: bool
    actual_return_time: Optional[float] = None
    actual_quantity: Optional[float] = None
    actual_duration: Optional[float] = None


class CalibrationTracker:
    """
    Tracks forecast outcomes against predictions grouped by tenant and agent_class (§8.1).
    """

    def __init__(self):
        # Key: (tenant_id_str, agent_class) -> List[ForecastOutcomePair]
        self._history: Dict[Tuple[str, str], List[ForecastOutcomePair]] = {}

    def record_outcome(
        self,
        forecast: DemandForecast,
        actual_materialized: bool,
        actual_return_time: Optional[float] = None,
        actual_quantity: Optional[float] = None,
        actual_duration: Optional[float] = None,
    ) -> ForecastOutcomePair:
        key = (str(forecast.tenant_id), forecast.agent_class)
        if key not in self._history:
            self._history[key] = []

        pair = ForecastOutcomePair(
            forecast_id=forecast.id,
            workflow_id=forecast.workflow_id,
            tenant_id=forecast.tenant_id,
            agent_class=forecast.agent_class,
            predicted_probability=forecast.probability,
            predicted_time_to_need_q50=forecast.time_to_need.q50,
            predicted_quantity_q50=forecast.quantity.q50,
            predicted_duration_q50=forecast.duration.q50,
            actual_materialized=actual_materialized,
            actual_return_time=actual_return_time,
            actual_quantity=actual_quantity,
            actual_duration=actual_duration,
        )

        self._history[key].append(pair)
        return pair

    def compute_calibration_score(self, tenant_id: UUID, agent_class: str) -> float:
        """
        Computes a calibration score in [0.0, 1.0] for a given (tenant, agent_class).
        Higher score means better accuracy/calibration.
        Combines materialization brier/accuracy error and return-time relative error.
        """
        key = (str(tenant_id), agent_class)
        history = self._history.get(key, [])
        if not history:
            return 1.0  # Conservative default for new classes/tenants (§8.3)

        probability_errors = []
        return_time_errors = []

        for p in history:
            # Materialization outcome: 1.0 if true, 0.0 if false
            actual_mat_val = 1.0 if p.actual_materialized else 0.0
            # Squared error for probability calibration
            p_err = (p.predicted_probability - actual_mat_val) ** 2
            probability_errors.append(p_err)

            if p.actual_materialized and p.actual_return_time is not None:
                # Relative error on time to need
                denom = max(1e-3, p.actual_return_time)
                t_err = abs(p.predicted_time_to_need_q50 - p.actual_return_time) / denom
                return_time_errors.append(min(1.0, t_err))

        mean_p_error = sum(probability_errors) / len(probability_errors) if probability_errors else 0.0
        mean_t_error = sum(return_time_errors) / len(return_time_errors) if return_time_errors else 0.0

        # Overall error composite (weighted)
        composite_error = 0.6 * mean_p_error + 0.4 * mean_t_error
        score = max(0.0, min(1.0, 1.0 - composite_error))
        return score
