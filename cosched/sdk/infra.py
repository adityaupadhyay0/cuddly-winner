from typing import Optional
from cosched.protocol.infra import InfraState
from cosched.protocol.relations import ResourceType


class CoSchedInfra:
    """
    SDK interface for local reading of InfraState (§12.1).
    Avoids RPC on local hot-path (NFR-02).
    """

    def __init__(self, daemon_client: Optional[object] = None):
        self.daemon_client = daemon_client
        self._local_cache: dict[str, InfraState] = {}

    def update_cached_state(self, state: InfraState):
        self._local_cache[state.resource.value] = state

    def state(self, resource: str = "GPU", current_time: Optional[float] = None) -> Optional[InfraState]:
        st = self._local_cache.get(resource)
        if st is None:
            return None
        if current_time is not None and st.valid_until <= current_time:
            del self._local_cache[resource]
            return None
        return st


infra = CoSchedInfra()
