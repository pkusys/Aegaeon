"""Compare paired baseline and Work Stealing GPU benchmark results."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


PERFORMANCE_METRICS = (
    "workload_makespan_seconds",
    "output_throughput_tokens_per_second",
    "mean_slo_attainment",
    "median_ttft_seconds",
    "p95_ttft_seconds",
    "median_tpot_seconds",
    "p95_tpot_seconds",
)

CV_METRICS = (
    "output_tokens",
    "active_seconds",
    "decode_steps",
    "turns_completed",
    "model_switches",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("work_stealing", type=Path)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        dest="output_format",
    )
    return parser.parse_args()


def load_result(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def _request_signature(result: Dict[str, Any]) -> tuple:
    return tuple(
        (
            request["request_id"],
            request["model"],
            request["input_tokens"],
            request["output_tokens"],
        )
        for request in result["requests"]
    )


def _validate_decode_token_conservation(
    name: str, result: Dict[str, Any]
) -> None:
    engine_tokens = sum(
        engine["output_tokens"] for engine in result["decode_load"]["engines"]
    )
    # Aegaeon produces the first generated token in Prefill. The Decode Engines
    # therefore produce total output tokens minus one token per request.
    expected_decode_tokens = result["output_tokens"] - result["request_count"]
    if engine_tokens != expected_decode_tokens:
        raise ValueError(
            f"{name}: Decode Engine tokens do not balance: "
            f"actual={engine_tokens}, expected={expected_decode_tokens}"
        )


def validate_pair(
    baseline: Dict[str, Any], work_stealing: Dict[str, Any]
) -> None:
    if baseline.get("setting") != "baseline":
        raise ValueError("first result is not a baseline run")
    if work_stealing.get("setting") != "work_stealing":
        raise ValueError("second result is not a Work Stealing run")

    for key in ("request_count", "output_tokens", "output_lengths"):
        if baseline.get(key) != work_stealing.get(key):
            raise ValueError(f"paired runs disagree on {key}")
    if _request_signature(baseline) != _request_signature(work_stealing):
        raise ValueError("paired runs contain different requests")

    baseline_engines = {
        engine["engine_id"] for engine in baseline["decode_load"]["engines"]
    }
    work_stealing_engines = {
        engine["engine_id"]
        for engine in work_stealing["decode_load"]["engines"]
    }
    if baseline_engines != work_stealing_engines:
        raise ValueError("paired runs contain different Decode Engines")

    _validate_decode_token_conservation("baseline", baseline)
    _validate_decode_token_conservation("work_stealing", work_stealing)


def _metric_delta(baseline: float, work_stealing: float) -> Dict[str, float]:
    relative_delta_percent = 0.0
    if baseline != 0:
        relative_delta_percent = (work_stealing / baseline - 1) * 100
    return {
        "baseline": baseline,
        "work_stealing": work_stealing,
        "absolute_delta": work_stealing - baseline,
        "relative_delta_percent": relative_delta_percent,
    }


def compare_results(
    baseline: Dict[str, Any], work_stealing: Dict[str, Any]
) -> Dict[str, Any]:
    validate_pair(baseline, work_stealing)
    comparison = {
        "performance": {
            metric: _metric_delta(baseline[metric], work_stealing[metric])
            for metric in PERFORMANCE_METRICS
        },
        "load_cv": {
            metric: _metric_delta(
                baseline["decode_load"]["cv"][metric],
                work_stealing["decode_load"]["cv"][metric],
            )
            for metric in CV_METRICS
        },
        "engines": {
            "baseline": baseline["decode_load"]["engines"],
            "work_stealing": work_stealing["decode_load"]["engines"],
        },
        "work_stealing": work_stealing["work_stealing"],
    }
    slo = comparison["performance"]["mean_slo_attainment"]
    slo["percentage_point_delta"] = slo["absolute_delta"] * 100
    return comparison


def _format_rows(
    title: str, metrics: Dict[str, Dict[str, float]], names: Iterable[str]
) -> list[str]:
    rows = [
        f"## {title}",
        "",
        "| Metric | Baseline | Work Stealing | Delta | Relative |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        metric = metrics[name]
        if name == "mean_slo_attainment":
            baseline = f"{metric['baseline'] * 100:.4f}%"
            work_stealing = f"{metric['work_stealing'] * 100:.4f}%"
            delta = f"{metric['percentage_point_delta']:+.4f} pp"
        else:
            baseline = f"{metric['baseline']:.6f}"
            work_stealing = f"{metric['work_stealing']:.6f}"
            delta = f"{metric['absolute_delta']:+.6f}"
        rows.append(
            f"| {name} | {baseline} | {work_stealing} | {delta} | "
            f"{metric['relative_delta_percent']:+.2f}% |"
        )
    return rows


def format_markdown(comparison: Dict[str, Any]) -> str:
    rows = _format_rows(
        "Performance", comparison["performance"], PERFORMANCE_METRICS
    )
    rows.extend(("",))
    rows.extend(_format_rows("Decode load CV", comparison["load_cv"], CV_METRICS))
    rows.extend(
        (
            "",
            "## Per-engine load",
            "",
            "| Setting | Engine | Decode tokens | Active seconds | "
            "Model switches | Donated | Received |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for setting in ("baseline", "work_stealing"):
        for engine in comparison["engines"][setting]:
            rows.append(
                f"| {setting} | {engine['engine_id']} | "
                f"{engine['output_tokens']} | {engine['active_seconds']:.6f} | "
                f"{engine['model_switches']} | "
                f"{engine['work_steals_donated']} | "
                f"{engine['work_steals_received']} |"
            )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    comparison = compare_results(
        load_result(args.baseline), load_result(args.work_stealing)
    )
    if args.output_format == "json":
        print(json.dumps(comparison, indent=2, sort_keys=True))
    else:
        print(format_markdown(comparison))


if __name__ == "__main__":
    main()
