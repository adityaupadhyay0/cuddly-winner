def compute_staleness_weight(issued_at: float, valid_until: float, now: float) -> float:
    """
    Computes linear staleness weight according to CoSched v0.3 SRS §5.5:

    age = now - issued_at
    window = valid_until - issued_at
    staleness_weight = clamp(1 - age / window, 0, 1)
    """
    window = valid_until - issued_at
    if window <= 0:
        return 0.0

    age = now - issued_at
    if age < 0:
        # Issued in future relative to now (clock skew guard)
        return 1.0

    raw_weight = 1.0 - (age / window)
    return max(0.0, min(1.0, raw_weight))


def compute_effective_probability(
    probability: float,
    issued_at: float,
    valid_until: float,
    now: float,
    calibration_weight: float = 1.0,
) -> float:
    """
    Computes effective probability (§5.5):
    effective_probability = probability * staleness_weight * calibration_weight
    """
    s_weight = compute_staleness_weight(issued_at, valid_until, now)
    return max(0.0, min(1.0, probability * s_weight * calibration_weight))
