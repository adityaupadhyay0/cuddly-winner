import time
import pytest
from uuid import uuid4
from cosched.sdk.ctx import CoSchedContext
from cosched.sdk.infra import CoSchedInfra
from cosched.daemon.daemon import ReferenceDaemon
from cosched.protocol.relations import ResourceType
from cosched.protocol.infra import InfraState
from cosched.protocol.distribution import Distribution


def test_sdk_and_daemon_integration():
    daemon = ReferenceDaemon()
    sdk_ctx = CoSchedContext(daemon_client=daemon)
    sdk_infra = CoSchedInfra(daemon_client=daemon)

    wf_id = uuid4()
    tenant_id = uuid4()
    now = 100.0

    # SDK forecast emission
    forecast = sdk_ctx.forecast(
        workflow_id=wf_id,
        agent_class="code_agent",
        tenant_id=tenant_id,
        resource="GPU",
        probability=0.85,
        time_to_need={"q50": 3.0, "q90": 5.0, "q99": 8.0},
        quantity={"q50": 2.0, "q90": 3.0, "q99": 4.0},
        duration={"q50": 1.0, "q90": 1.5, "q99": 2.0},
        issued_at=now,
        valid_for=10.0,
    )

    assert forecast.workflow_id == wf_id

    # Publish InfraState
    dist = Distribution(q50=10.0, q90=10.0, q99=10.0)
    state = InfraState(
        resource=ResourceType.GPU,
        capacity=dist,
        availability=dist,
        expected_wait=dist,
        issued_at=now,
        valid_until=now + 5.0,
    )
    daemon.publish_infra_state(state, now=now)
    sdk_infra.update_cached_state(state)

    assert sdk_infra.state("GPU", current_time=now + 1.0) is not None

    # Daemon tick
    decisions = daemon.tick(now=now + 1.0)
    assert len(decisions) == 1
    assert decisions[0].workflow_id == str(wf_id)
