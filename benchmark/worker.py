import json


def compute_request_slo(per_token, ttft_slo: float, tpot_slo: float) -> float:
    """Compute the per-request time-based SLO attainment used by Aegaeon."""
    if not per_token:
        return 0.0
    qos = 0.0
    start = 0.0
    accum = 0.0
    target = ttft_slo
    sum_token = 0.0
    for token in per_token:
        sum_token += token
        if sum_token - start <= target:
            accum = target
            target += tpot_slo
        else:
            qos += accum
            start = sum_token
            accum = 0.0
            target += tpot_slo
    total = sum(per_token)
    qos += min(target, total - start)
    return qos / total if total > 0 else 0.0


def compute_slo_from_metrics(req_metrics, ttft_slo: float, tpot_slo: float) -> float:
    values = [
        compute_request_slo(metrics.get("per_token", []), ttft_slo, tpot_slo)
        for metrics in req_metrics.values()
    ]
    return sum(values) / len(values) if values else 0.0


def compute_slo(
    json_file,
    ttft_slo: float,
    tpot_slo: float,
) -> float:

    # Compute SLO Attainment
    TTFT_SLO = ttft_slo
    TPOT_SLO = tpot_slo

    with open(json_file, "r") as f:
        req_metrics = json.load(f)

    # QoS is defined by the pecentage of time where TTFT/TPOT requirements
    # are satisfied over total service time.
    # For example, a request arriving at 0 with each token time (3, 6, 7, 8)
    # and (TTFT_SLO, TPOT_SLO) = (3, 2) has QoS 5/8 from interval (0, 3) and
    # (6, 8).

    return compute_slo_from_metrics(req_metrics, TTFT_SLO, TPOT_SLO)
