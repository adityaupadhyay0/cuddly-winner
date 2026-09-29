import pytest
from uuid import uuid4
from cosched.calibration.staleness import compute_staleness_weight, compute_effective_probability
from cosched.calibration.calibration import CalibrationTracker
from cosched.calibration.trust import TrustModel
from cosched.protocol.demand import DemandForecast
from cosched.protocol.distribution import Distribution
from cosched.protocol.relations import ResourceType


def test_staleness_weighting():
    # Fresh message: age = 0
    w0 = compute_staleness_weight(issued_at=10.0, valid_until=20.0, now=10.0)
    assert w0 == 1.0

    # Halfway expired: age = 5
    w5 = compute_staleness_weight(issued_at=10.0, valid_until=20.0, now=15.0)
    assert w5 == 0.5

    # Expired: age = 10
    w10 = compute_staleness_weight(issued_at=10.0, valid_until=20.0, now=20.0)
    assert w10 == 0.0

    # Effective probability
    eff = compute_effective_probability(probability=0.8, issued_at=10.0, valid_until=20.0, now=15.0, calibration_weight=0.9)
    assert pytest.approx(eff) == 0.8 * 0.5 * 0.9


def test_calibration_and_trust():
    tracker = CalibrationTracker()
    trust_model = TrustModel(floor=0.2)

    tenant_id = uuid4()
    agent_class = "code_agent"

    # Default trust before data
    initial_trust = trust_model.compute_trust_weight(tracker, tenant_id, agent_class)
    assert initial_trust == 1.0

    dist = Distribution(q50=3.0, q90=5.0, q99=8.0)
    forecast = DemandForecast(
        workflow_id=uuid4(),
        agent_class=agent_class,
        tenant_id=tenant_id,
        resource=ResourceType.GPU,
        quantity=dist,
        time_to_need=dist,
        duration=dist,
        probability=0.9,
        issued_at=0.0,
        revision=1,
        valid_until=10.0,
    )

    # Record perfectly accurate outcomes
    for _ in range(5):
        tracker.record_outcome(
            forecast=forecast,
            actual_materialized=True,
            actual_return_time=3.0,
            actual_quantity=3.0,
            actual_duration=3.0,
        )

    high_score_trust = trust_model.compute_trust_weight(tracker, tenant_id, agent_class)
    assert high_score_trust >= 0.9

    # Record severe over-claims / inaccuracies (predicted prob=0.9, but never materializes)
    for _ in range(20):
        tracker.record_outcome(
            forecast=forecast,
            actual_materialized=False,
        )

    low_score_trust = trust_model.compute_trust_weight(tracker, tenant_id, agent_class)
    assert low_score_trust < high_score_trust
    assert low_score_trust >= 0.2  # Should not fall below floor
