import pytest
from uuid import uuid4
from cosched.protocol.distribution import Distribution
from cosched.protocol.relations import ResourceType, RelationSet, PressureVector
from cosched.protocol.demand import DemandForecast
from cosched.protocol.infra import InfraState
from cosched.protocol.lifecycle import LifecycleManager


def test_distribution_validation():
    # Monotonic valid distribution
    d = Distribution(q50=3.0, q90=5.0, q99=8.0, min=1.0, max=10.0)
    assert d.q50 == 3.0
    assert d.interpolate_quantile(0.5) == 3.0

    # Non-monotonic quantiles should fail
    with pytest.raises(ValueError, match="Quantiles must be monotonic"):
        Distribution(q50=5.0, q90=3.0, q99=8.0)

    # Min higher than q50 should fail
    with pytest.raises(ValueError, match="min .* must be <= q50"):
        Distribution(q50=3.0, q90=5.0, q99=8.0, min=4.0)

    # Max lower than q99 should fail
    with pytest.raises(ValueError, match="max .* must be >= q99"):
        Distribution(q50=3.0, q90=5.0, q99=8.0, max=7.0)


def test_forecast_creation_and_lifecycle():
    wf_id = uuid4()
    tenant_id = uuid4()
    forecast_id = uuid4()

    dist = Distribution(q50=1.0, q90=2.0, q99=3.0)

    f1 = DemandForecast(
        id=forecast_id,
        workflow_id=wf_id,
        agent_class="code_agent",
        tenant_id=tenant_id,
        resource=ResourceType.GPU,
        quantity=dist,
        time_to_need=dist,
        duration=dist,
        probability=0.85,
        issued_at=10.0,
        revision=1,
        valid_until=20.0,
    )

    manager = LifecycleManager()

    # Ingest revision 1
    accepted, reason = manager.ingest_forecast(f1, current_time=11.0)
    assert accepted
    assert reason == "accepted"

    active = manager.get_active_forecasts(current_time=12.0)
    assert forecast_id in active

    # Try ingesting late / lower revision number
    f_late = f1.model_copy(update={"revision": 1, "issued_at": 11.0})
    accepted, reason = manager.ingest_forecast(f_late, current_time=12.0)
    assert not accepted
    assert "late_or_duplicate_revision" in reason

    # Ingest newer revision
    f2 = f1.model_copy(update={"revision": 2, "probability": 0.95, "issued_at": 12.0})
    accepted, reason = manager.ingest_forecast(f2, current_time=13.0)
    assert accepted
    assert manager.superseded_count == 1

    active = manager.get_active_forecasts(current_time=14.0)
    assert active[forecast_id].probability == 0.95

    # Check expiry
    active_after_expiry = manager.get_active_forecasts(current_time=21.0)
    assert forecast_id not in active_after_expiry


def test_infra_state_lifecycle():
    dist = Distribution(q50=10.0, q90=10.0, q99=10.0)
    state = InfraState(
        resource=ResourceType.GPU,
        capacity=dist,
        availability=dist,
        expected_wait=dist,
        pressure=PressureVector(gpu=0.8, kv=0.5),
        issued_at=100.0,
        valid_until=105.0,
    )

    manager = LifecycleManager()

    # Ingest state
    accepted, reason = manager.ingest_infra_state(state, current_time=101.0)
    assert accepted

    active = manager.get_active_infra_state("GPU", current_time=102.0)
    assert active is not None
    assert active.pressure.gpu == 0.8

    # Expired check
    expired_state = manager.get_active_infra_state("GPU", current_time=106.0)
    assert expired_state is None
