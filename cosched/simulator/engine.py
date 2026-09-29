import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from cosched.simulator.traces import AgentWorkloadTrace, StepEvent
from cosched.simulator.metrics import SimulationMetrics
from cosched.protocol.demand import DemandForecast
from cosched.protocol.infra import InfraState
from cosched.protocol.distribution import Distribution
from cosched.protocol.relations import ResourceType, PressureVector
from cosched.calibration.calibration import CalibrationTracker
from cosched.calibration.trust import TrustModel
from cosched.policies.kv_retention import decide_kv_retention


@dataclass(order=True)
class Event:
    time: float
    event_type: str = field(compare=False)
    data: Any = field(compare=False)


class SimulationEngine:
    """
    Discrete-event simulator for CoSched v0.3 (§11).
    Supports policy evaluations P0, P1, P2, P3, P4, and Oracle (§11.4).
    """

    def __init__(
        self,
        policy: str = "P0",
        num_gpus: int = 4,
        max_kv_capacity_gb: float = 32.0,
        forecast_horizon: float = 30.0,
        telemetry_delay: float = 0.05,
        forecast_error_std: float = 0.0,
        seed: int = 42,
    ):
        self.policy = policy.upper()
        self.num_gpus = num_gpus
        self.max_kv_capacity_gb = max_kv_capacity_gb
        self.forecast_horizon = forecast_horizon
        self.telemetry_delay = telemetry_delay
        self.forecast_error_std = forecast_error_std
        self.rng = random.Random(seed)

        self.current_time = 0.0
        self.event_queue: List[Event] = []

        # Resources state
        self.active_gpu_jobs: int = 0
        self.occupied_kv_gb: float = 0.0
        # Active KV blocks by workflow_id: wf_id -> kv_gb
        self.kv_store: Dict[UUID, float] = {}
        # Track workflows needing recomputation on next inference
        self.pending_recompute: Dict[UUID, float] = {}

        # Workload queues
        self.gpu_queue: List[Tuple[AgentWorkloadTrace, int, float]] = []

        # Calibration
        self.calibration_tracker = CalibrationTracker()
        self.trust_model = TrustModel(floor=0.2)

        # Metrics
        self.metrics = SimulationMetrics()

    def schedule_event(self, time: float, event_type: str, data: Any):
        heapq.heappush(self.event_queue, Event(time=time, event_type=event_type, data=data))

    def run(self, traces: List[AgentWorkloadTrace], max_time: float = 500.0) -> SimulationMetrics:
        self.metrics.total_workflows = len(traces)

        # Schedule initial workflow starts
        for trace in traces:
            self.schedule_event(0.0, "WORKFLOW_START", {"trace": trace, "step_idx": 0, "start_time": 0.0})

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            if event.time > max_time:
                break

            # Update simulation time
            dt = event.time - self.current_time
            if self.active_gpu_jobs > 0:
                self.metrics.total_gpu_active_time += dt * self.active_gpu_jobs
            else:
                self.metrics.total_gpu_idle_time += dt * self.num_gpus

            self.current_time = event.time
            self._handle_event(event)

        self.metrics.total_sim_time = self.current_time
        return self.metrics

    def _handle_event(self, event: Event):
        e_type = event.event_type
        data = event.data

        if e_type == "WORKFLOW_START":
            self._process_step(data["trace"], data["step_idx"], data["start_time"])

        elif e_type == "INFERENCE_DONE":
            trace = data["trace"]
            step_idx = data["step_idx"]
            start_time = data["start_time"]
            self.active_gpu_jobs -= 1

            # Move to next step
            self._process_step(trace, step_idx + 1, start_time)
            self._try_schedule_gpu()

        elif e_type == "TOOL_WAIT_DONE":
            trace = data["trace"]
            step_idx = data["step_idx"]
            start_time = data["start_time"]
            wf_id = trace.workflow_id

            # Check if KV cache was evicted
            if wf_id not in self.kv_store:
                self.metrics.total_kv_recomputations += 1
                # Mark pending recompute cost for next inference
                step = trace.steps[step_idx]
                self.pending_recompute[wf_id] = step.recompute_cost

            # Move to next step
            self._process_step(trace, step_idx + 1, start_time)

    def _process_step(self, trace: AgentWorkloadTrace, step_idx: int, start_time: float):
        if step_idx >= len(trace.steps):
            self._finish_workflow(trace, start_time)
            return

        step = trace.steps[step_idx]

        if step.step_type == "DONE":
            self._finish_workflow(trace, start_time)

        elif step.step_type == "INFERENCE":
            # Allocate KV memory for workflow if evicted
            if trace.workflow_id not in self.kv_store:
                self.kv_store[trace.workflow_id] = step.kv_size_gb
                self.occupied_kv_gb += step.kv_size_gb

            # Queue or run inference
            self.gpu_queue.append((trace, step_idx, start_time))
            self._try_schedule_gpu()

        elif step.step_type == "TOOL_WAIT":
            wf_id = trace.workflow_id
            kv_size = self.kv_store.get(wf_id, step.kv_size_gb)

            # Determine whether to retain or evict based on active Policy
            should_retain = self._evaluate_kv_policy(trace, step)

            if not should_retain and wf_id in self.kv_store:
                # Evict KV Cache
                del self.kv_store[wf_id]
                self.occupied_kv_gb -= kv_size
                self.metrics.total_kv_evictions += 1

            # Schedule completion of tool wait
            self.schedule_event(
                self.current_time + step.duration,
                "TOOL_WAIT_DONE",
                {"trace": trace, "step_idx": step_idx, "start_time": start_time},
            )

    def _evaluate_kv_policy(self, trace: AgentWorkloadTrace, step: StepEvent) -> bool:
        """
        Evaluates KV retain/evict decision according to active Policy (§11.4).
        """
        if self.policy == "P0":
            kv_pressure = self.occupied_kv_gb / self.max_kv_capacity_gb
            return kv_pressure < 0.5

        elif self.policy == "ORACLE":
            return step.duration <= 5.0

        elif self.policy in ("P1", "P2", "P3", "P4"):
            time_to_need = step.duration
            if self.forecast_error_std > 0.0:
                time_to_need += self.rng.gauss(0.0, self.forecast_error_std)
                time_to_need = max(0.1, time_to_need)

            # Omit forecast if beyond horizon (§5.6)
            if time_to_need > self.forecast_horizon:
                # Fall back to baseline P0 behavior
                kv_pressure = self.occupied_kv_gb / self.max_kv_capacity_gb
                return kv_pressure < 0.5

            prob = step.probability

            if self.policy == "P1":
                time_to_need = time_to_need * 1.1
                prob = prob * 0.9

            elif self.policy == "P2":
                self.metrics.total_forecasts_sent += 1
                trust_w = self.trust_model.compute_trust_weight(self.calibration_tracker, trace.tenant_id, trace.agent_class)
                prob = prob * trust_w

            elif self.policy == "P3":
                # P3: Closed-loop InfraState feedback
                self.metrics.total_forecasts_sent += 1
                trust_w = self.trust_model.compute_trust_weight(self.calibration_tracker, trace.tenant_id, trace.agent_class)
                prob = prob * trust_w
                # Closed loop adjusts decision threshold if telemetry pressure is high
                kv_pressure = self.occupied_kv_gb / self.max_kv_capacity_gb
                if kv_pressure > 0.8:
                    prob *= 0.8  # Closed loop structural hint adaptation

            elif self.policy == "P4":
                # P4: Coordination group relationship policy
                self.metrics.total_forecasts_sent += 1
                trust_w = self.trust_model.compute_trust_weight(self.calibration_tracker, trace.tenant_id, trace.agent_class)
                prob = prob * trust_w
                if trace.coordination_group:
                    # Boost priority/retention probability for coordinated swarm tasks
                    prob = min(1.0, prob * 1.15)

            kv_pressure = self.occupied_kv_gb / self.max_kv_capacity_gb
            opp_cost_rate = 0.5 * (1.0 + kv_pressure)

            decision = decide_kv_retention(
                workflow_id=str(trace.workflow_id),
                kv_size_gb=step.kv_size_gb,
                expected_hold_time=time_to_need,
                effective_probability=prob,
                gpu_memory_opportunity_cost_per_gb_sec=opp_cost_rate,
                recompute_or_reload_cost=step.recompute_cost,
            )

            # Record calibration
            d = Distribution(q50=time_to_need, q90=time_to_need * 1.2, q99=time_to_need * 1.5)
            fc = DemandForecast(
                workflow_id=trace.workflow_id,
                agent_class=trace.agent_class,
                tenant_id=trace.tenant_id,
                resource=ResourceType.GPU,
                quantity=d,
                time_to_need=d,
                duration=d,
                probability=step.probability,
                issued_at=self.current_time,
                revision=1,
                valid_until=self.current_time + self.forecast_horizon,
            )
            self.calibration_tracker.record_outcome(
                fc, actual_materialized=True, actual_return_time=step.duration
            )

            return decision.retain

        return True

    def _try_schedule_gpu(self):
        while self.gpu_queue and self.active_gpu_jobs < self.num_gpus:
            trace, step_idx, start_time = self.gpu_queue.pop(0)
            step = trace.steps[step_idx]

            # Calculate actual inference duration, adding recompute penalty if KV was evicted
            recompute_penalty = self.pending_recompute.pop(trace.workflow_id, 0.0)
            actual_duration = step.duration + recompute_penalty

            self.active_gpu_jobs += 1
            self.schedule_event(
                self.current_time + actual_duration,
                "INFERENCE_DONE",
                {"trace": trace, "step_idx": step_idx, "start_time": start_time},
            )

    def _finish_workflow(self, trace: AgentWorkloadTrace, start_time: float):
        jct = self.current_time - start_time
        self.metrics.completed_workflows += 1
        self.metrics.job_completion_times.append(jct)

        tenant_key = str(trace.tenant_id)
        self.metrics.tenant_completed[tenant_key] = self.metrics.tenant_completed.get(tenant_key, 0) + 1

        if trace.deadline and jct > trace.deadline:
            self.metrics.deadline_misses += 1

        if trace.workflow_id in self.kv_store:
            self.occupied_kv_gb -= self.kv_store[trace.workflow_id]
            del self.kv_store[trace.workflow_id]
