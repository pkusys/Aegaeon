"""Run a CSV workload against the OpenAI-compatible HTTP API."""

import argparse
import asyncio
import csv
import math
import time
from dataclasses import dataclass
from typing import Optional


DEFAULT_MODELS = (
    "qwen2_5_7b",
    "qwen2_1_5b",
    "llama-3-8b",
    "mistral-7b-v0.1",
    "mixtral-8x7b",
    "gemma-7b",
)
DEFAULT_REQUEST_TIMEOUT = 300.0


def positive_finite_timeout(value):
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("request timeout must be a finite positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("request timeout must be a finite positive number")
    return timeout


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: int
    csv_row: int
    model: str
    prompt: str
    output_tokens: int
    arrival_seconds: float


@dataclass(frozen=True)
class RequestResult:
    request_id: Optional[int]
    csv_row: Optional[int]
    model: str
    status: str
    latency: float
    http_status: Optional[int] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkResult:
    results: list[RequestResult]
    elapsed: float

    @property
    def success(self):
        return sum(result.status == "success" for result in self.results)

    @property
    def failed(self):
        return len(self.results) - self.success

    @property
    def timeouts(self):
        return sum(result.status == "timeout" for result in self.results)

    @property
    def exit_code(self):
        return int(self.failed > 0)


def load_workload(csv_path, models=DEFAULT_MODELS):
    if not models:
        raise ValueError("at least one model is required")
    requests = []
    with open(csv_path, "r", newline="") as source:
        reader = csv.reader(source)
        next(reader)  # Header
        for request_id, row in enumerate(reader):
            csv_row = reader.line_num
            try:
                req_time_str, raw_model_name, in_len_str, out_len_str = row
                h, m, s_ms = req_time_str.split(":")
                s, ms = s_ms.split(".")
                arrival_seconds = (
                    int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                )
                try:
                    model_id = int(raw_model_name.split("_")[-1])
                except ValueError:
                    model_id = 0
                model_name = models[model_id % len(models)]
                input_tokens = int(in_len_str)
                output_tokens = int(out_len_str)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid workload CSV row {csv_row}: {row}") from exc

            # Keep the workload's original placeholder prompt construction.
            requests.append(WorkloadRequest(
                request_id=request_id,
                csv_row=csv_row,
                model=model_name,
                prompt="test " * input_tokens,
                output_tokens=output_tokens,
                arrival_seconds=arrival_seconds,
            ))
    return requests


async def send_request(
    session, url, model_name, dummy_prompt, out_len, delay_seconds,
    request_timeout=DEFAULT_REQUEST_TIMEOUT, *, request_id=None, csv_row=None,
):
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=positive_finite_timeout(request_timeout))
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": dummy_prompt}],
        "max_tokens": out_len,
        "stream": False,
    }
    identity = f"Request: {request_id} | CSV row: {csv_row} | Model: {model_name}"
    start_time = time.monotonic()
    try:
        # The request timeout starts here, after the workload arrival delay.
        async with session.post(url, json=payload, timeout=timeout) as response:
            if response.status == 200:
                await response.json()
                result = RequestResult(
                    request_id, csv_row, model_name, "success",
                    time.monotonic() - start_time, response.status,
                )
                print(
                    f"[Success] {identity} | Latency: {result.latency:.2f}s | "
                    f"Target Output Tokens: {out_len}"
                )
                return result

            detail = await response.text()
            result = RequestResult(
                request_id, csv_row, model_name, "failed",
                time.monotonic() - start_time, response.status, detail,
            )
            print(f"[Failed] {identity} | Status: {response.status} | Detail: {detail}")
            return result
    except asyncio.TimeoutError:
        detail = f"Request exceeded timeout of {timeout.total:g}s"
        result = RequestResult(
            request_id, csv_row, model_name, "timeout",
            time.monotonic() - start_time, detail=detail,
        )
        print(f"[Timeout] {identity} | Detail: {detail}")
        return result
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        result = RequestResult(
            request_id, csv_row, model_name, "failed",
            time.monotonic() - start_time, detail=detail,
        )
        print(f"[Failed] {identity} | Detail: {detail}")
        return result


async def main(args):
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    timeout = positive_finite_timeout(
        getattr(args, "request_timeout", DEFAULT_REQUEST_TIMEOUT)
    )
    models = getattr(args, "models", None) or DEFAULT_MODELS
    print(f"Reading Workload file: {args.csv}")
    requests = load_workload(args.csv, models)
    print(f"Starting benchmark, loaded {len(requests)} concurrent requests...\n")
    import aiohttp

    global_start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(send_request(
                session, url, request.model, request.prompt,
                request.output_tokens, request.arrival_seconds, timeout,
                request_id=request.request_id, csv_row=request.csv_row,
            ))
            for request in requests
        ]
        results = await asyncio.gather(*tasks)
    benchmark = BenchmarkResult(results, time.monotonic() - global_start)
    print(
        f"\nBenchmark finished! Total time: {benchmark.elapsed:.2f}s\n"
        f"Summary: total={len(results)} success={benchmark.success} "
        f"failed={benchmark.failed} timeouts={benchmark.timeouts}"
    )
    return benchmark


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to the Workload CSV file")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--request-timeout", type=positive_finite_timeout,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Maximum seconds per HTTP request after its arrival delay (default: 300)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Override the ordered model aliases used for synth_model_00, synth_model_01, etc.",
    )
    return parser


def run_cli(argv=None):
    args = build_parser().parse_args(argv)
    return asyncio.run(main(args)).exit_code


if __name__ == "__main__":
    raise SystemExit(run_cli())
