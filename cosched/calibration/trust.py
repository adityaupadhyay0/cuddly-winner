from uuid import UUID
from cosched.calibration.calibration import CalibrationTracker


class TrustModel:
    """
    Computes trust weighting according to CoSched v0.3 §8.2:

    calibration_weight = floor + (1 - floor) * calibration_score
    """

    def __init__(self, floor: float = 0.2):
        if not (0.0 <= floor <= 1.0):
            raise ValueError("Floor must be between 0.0 and 1.0")
        self.floor = floor

    def compute_trust_weight(self, tracker: CalibrationTracker, tenant_id: UUID, agent_class: str) -> float:
        score = tracker.compute_calibration_score(tenant_id, agent_class)
        weight = self.floor + (1.0 - self.floor) * score
        return max(self.floor, min(1.0, weight))
