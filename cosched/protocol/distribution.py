from typing import Optional
from pydantic import BaseModel, Field, model_validator


class Distribution(BaseModel):
    """
    Quantile representation of a uncertainty distribution in CoSched v0.3 (§4.3).
    Values must satisfy min <= q50 <= q90 <= q99 <= max when bounds are provided.
    """
    q50: float = Field(ge=0.0, description="50th percentile / median estimate")
    q90: float = Field(ge=0.0, description="90th percentile estimate")
    q99: float = Field(ge=0.0, description="99th percentile estimate")
    min: Optional[float] = Field(default=None, ge=0.0, description="Known lower bound")
    max: Optional[float] = Field(default=None, ge=0.0, description="Known upper bound")

    @model_validator(mode="after")
    def check_monotonic_quantiles(self) -> "Distribution":
        q50, q90, q99 = self.q50, self.q90, self.q99
        min_val, max_val = self.min, self.max

        if not (q50 <= q90 <= q99):
            raise ValueError(f"Quantiles must be monotonic: q50 ({q50}) <= q90 ({q90}) <= q99 ({q99})")

        if min_val is not None:
            if min_val > q50:
                raise ValueError(f"min ({min_val}) must be <= q50 ({q50})")

        if max_val is not None:
            if max_val < q99:
                raise ValueError(f"max ({max_val}) must be >= q99 ({q99})")

        return self

    def interpolate_quantile(self, q: float) -> float:
        """
        Monotone piecewise-linear quantile function interpolation (§4.3).
        q is expected in [0, 1].
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError("Quantile q must be between 0.0 and 1.0")

        min_val = self.min if self.min is not None else 0.0
        max_val = self.max if self.max is not None else self.q99 * 1.5

        points = [
            (0.0, min_val),
            (0.5, self.q50),
            (0.9, self.q90),
            (0.99, self.q99),
            (1.0, max_val),
        ]

        # Handle boundary/duplicate points cleanly
        for i in range(len(points) - 1):
            q_low, v_low = points[i]
            q_high, v_high = points[i + 1]
            if q_low <= q <= q_high:
                if q_high == q_low:
                    return v_low
                fraction = (q - q_low) / (q_high - q_low)
                return v_low + fraction * (v_high - v_low)

        return max_val
