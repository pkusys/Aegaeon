import os
import asyncio
import aiohttp
import json
import time
import random
import csv
import argparse

DIR = os.path.dirname(__file__)

_PROMPT = (
    "Can you provide a detailed analysis of the main challenges in scaling "
    "transformer-based language models to process very long sequences, covering "
    "quadratic attention complexity, memory footprint, and inference latency? "
    "What are the most effective current approaches, such as sparse attention, "
    "linear transformers, or recurrent models like RWKV? How do they compare "
    "in accuracy and speed for tasks like book-length summarization or "
    "open-domain question answering? Also, discuss any practical trade-offs "
    "and implementation difficulties, and suggest potential future directions."
)
#_PROMPT = (
#    "The capital of China is "
#)


async def _consume_stream(resp, start, print_stream=False, label=""):
    """Consume an OpenAI-compatible SSE response as chunks arrive."""
    output_parts = []
    first_token_latency = None
    chunk_count = 0
    finish_reason = None

    if print_stream:
        print(f"{label}", end="", flush=True)

    # aiohttp's StreamReader iteration is line-oriented.  Processing each
    # `data:` line here avoids buffering the complete response in the client.
    async for raw_line in resp.content:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue

        event_data = line[len("data:"):].strip()
        if event_data == "[DONE]":
            break
        if not event_data:
            continue

        event = json.loads(event_data)
        choices = event.get("choices") or []
        if not choices:
            # Usage-only chunks have an empty choices list.
            continue

        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        if not text:
            continue

        if first_token_latency is None:
            first_token_latency = time.time() - start
        chunk_count += 1
        output_parts.append(text)
        if print_stream:
            print(text, end="", flush=True)

    if print_stream:
        print(flush=True)

    return {
        "output": "".join(output_parts),
        "ttft": first_token_latency,
        "chunks": chunk_count,
        "finish_reason": finish_reason,
    }


async def send_one(
    session,
    url,
    model_name,
    out_len,
    delay,
    stream=False,
    print_stream=False,
    request_index=0,
    ignore_eos=False,
):
    if delay > 0:
        await asyncio.sleep(delay)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": _PROMPT}],
        "max_tokens": out_len,
        "stream": stream,
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    start = time.time()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                if stream:
                    stream_result = await _consume_stream(
                        resp,
                        start,
                        print_stream=print_stream,
                        label=f"[request {request_index}] " if print_stream else "",
                    )
                    return {
                        "model": model_name,
                        "out_len": out_len,
                        "latency": time.time() - start,
                        "ok": True,
                        "tokens": stream_result["chunks"],
                        **stream_result,
                    }

                data = await resp.json()
                latency = time.time() - start
                generated = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                actual_tokens = data.get("usage", {}).get("completion_tokens", 0)
                return {
                    "model": model_name,
                    "out_len": out_len,
                    "latency": latency,
                    "output": generated,
                    "ok": True,
                    "tokens": actual_tokens,
                }
            else:
                error_body = await resp.text()
                return {"model": model_name, "out_len": out_len,
                        "latency": time.time() - start, "output": "", "ok": False,
                        "status": resp.status, "error": error_body}
    except Exception as e:
        return {"model": model_name, "out_len": out_len,
                "latency": time.time() - start, "output": "", "ok": False,
                "error": str(e)}


async def main(args):
    from aegaeon.models import set_model_registry
    set_model_registry(args.model_config)

    from aegaeon.model_registry import get_registry
    registry = get_registry()
    models = list(registry)

    if args.model:
        selected = registry.get_by_name(args.model)
        if selected is None:
            raise ValueError(
                f"Model '{args.model}' is not present in {args.model_config}"
            )
        models = [selected]

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    jobs = []

    if args.csv:
        with open(args.csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = int(row.get("request_id", 0))
                ts = float(row.get("timestamp", 0))
                input_tokens = int(row.get("input_tokens", 256))
                out_len = int(row.get("output_tokens", 128))
                model = models[rid % len(models)]
                jobs.append((model.name, out_len, ts))
    elif args.single:
        model = models[0]
        jobs.append((model.name, args.max_tokens, 0.0))
    else:
        rng = random.Random(args.seed)
        arrival = 0.0
        idx = 0
        while arrival < args.duration:
            arrival += rng.expovariate(args.arrival_rate)
            if arrival >= args.duration:
                break
            model = models[idx]
            out_len = 500
            jobs.append((model.name, out_len, arrival))
            idx = (idx + 1) % len(models)

    print(f"Starting API benchmark: {len(jobs)} requests")
    print(f"Target: {url}")
    print(f"Models: {[m.name for m in models]}")
    print(f"Streaming: {args.stream}")
    print(f"Ignore EOS: {args.ignore_eos}")
    print()

    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_one(
                session,
                url,
                name,
                ol,
                delay,
                stream=args.stream,
                print_stream=args.print_stream,
                request_index=index,
                ignore_eos=args.ignore_eos,
            )
            for index, (name, ol, delay) in enumerate(jobs)
        ]
        results = await asyncio.gather(*tasks)
    total = time.time() - start

    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    latencies = [r["latency"] for r in ok]

    print(f"Total: {len(results)}, OK: {len(ok)}, Fail: {len(fail)}")
    if ok:
        print(f"Latency: min={min(latencies):.2f}s avg={sum(latencies)/len(latencies):.2f}s max={max(latencies):.2f}s")
        tokens = [r.get("tokens", 0) for r in ok]
        print(f"Tokens:  min={min(tokens)} avg={sum(tokens)/len(tokens):.1f} max={max(tokens)}")
        reasons = [r.get("finish_reason", "?") for r in ok]
        from collections import Counter
        reason_counts = Counter(reasons)
        print(f"Finish: {dict(reason_counts)}")
        if args.stream:
            ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
            chunks = [r["chunks"] for r in ok]
            if ttfts:
                print(
                    "TTFT: "
                    f"min={min(ttfts):.2f}s "
                    f"avg={sum(ttfts)/len(ttfts):.2f}s "
                    f"max={max(ttfts):.2f}s"
                )
            print(
                "SSE chunks: "
                f"min={min(chunks)} "
                f"avg={sum(chunks)/len(chunks):.1f} "
                f"max={max(chunks)}"
            )
    print(f"Wall time: {total:.2f}s")

    out_path = os.path.join(DIR, "demo-output.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "total_time": total}, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--arrival-rate", type=float, default=0.1)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="", help="Path to CSV workload file")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--stream", action="store_true", help="Consume the response as SSE"
    )
    parser.add_argument(
        "--print-stream",
        action="store_true",
        help="Print each text delta immediately (best used with --single)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Send one request immediately instead of a timed workload",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Output length for --single",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Send ignore_eos=true so the backend does not stop at EOS",
    )
    args = parser.parse_args()
    if args.print_stream and not args.stream:
        parser.error("--print-stream requires --stream")
    asyncio.run(main(args))
