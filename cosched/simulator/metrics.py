from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimulationMetrics:
    total_workflows: int = 0
    completed_workflows: int = 0
    total_kv_evictions: int = 0
    total_kv_recomputations: int = 0
    total_gpu_active_time: float = 0.0
    total_gpu_idle_time: float = 0.0
    total_sim_time: float = 0.0
    job_completion_times: List[float] = field(default_factory=list)
    deadline_misses: int = 0
    total_forecasts_sent: int = 0
    tenant_completed: Dict[str, int] = field(default_factory=dict)

    @property
    def kv_eviction_rate(self) -> float:
        if self.completed_workflows == 0:
            return 0.0
        return self.total_kv_evictions / self.completed_workflows

    @property
    def recomputation_rate(self) -> float:
        if self.total_kv_evictions == 0:
            return 0.0
        return self.total_kv_recomputations / self.total_kv_evictions

    @property
    def jct_p50(self) -> float:
        if not self.job_completion_times:
            return 0.0
        s = sorted(self.job_completion_times)
        return s[int(len(s) * 0.5)]

    @property
    def jct_p95(self) -> float:
        if not self.job_completion_times:
            return 0.0
        s = sorted(self.job_completion_times)
        idx = min(len(s) - 1, int(len(s) * 0.95))
        return s[idx]

    @property
    def tenant_fairness_jain(self) -> float:
        """
        Jain's Fairness Index across tenant completion counts.
        """
        counts = list(self.tenant_completed.values())
        if not counts or sum(counts) == 0:
            return 1.0
        n = len(counts)
        sum_x = sum(counts)
        sum_sq_x = sum(x ** 2 for x in counts)
        if sum_sq_x == 0:
            return 1.0
        return (sum_x ** 2) / (n * sum_sq_x)

    def to_dict(self) -> dict:
        return {
            "total_workflows": self.total_workflows,
            "completed_workflows": self.completed_workflows,
            "kv_evictions": self.total_kv_evictions,
            "kv_recomputations": self.total_kv_recomputations,
            "kv_eviction_rate": round(self.kv_eviction_rate, 4),
            "recomputation_rate": round(self.recomputation_rate, 4),
            "gpu_active_time": round(self.total_gpu_active_time, 2),
            "gpu_idle_time": round(self.total_gpu_idle_time, 2),
            "jct_p50": round(self.jct_p50, 2),
            "jct_p95": round(self.jct_p95, 2),
            "deadline_misses": self.deadline_misses,
            "tenant_fairness": round(self.tenant_fairness_jain, 4),
            "forecasts_sent": self.total_forecasts_sent,
        }
