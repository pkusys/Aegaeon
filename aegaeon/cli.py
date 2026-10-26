import argparse
import http.client
import json
import sys


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show parser defaults while preserving command examples."""

    def _get_help_string(self, action):
        if action.required:
            return action.help
        return super()._get_help_string(action)


def main():
    parser = argparse.ArgumentParser(
        prog="aegaeon",
        description="Serve multiple language models with token-level GPU scheduling.",
        formatter_class=_HelpFormatter,
        epilog=(
            "Use 'aegaeon COMMAND --help' for all options of a command.\n\n"
            "Examples:\n"
            "  aegaeon doctor --check-cuda\n"
            "  aegaeon start --config aegaeon.yaml --ray-address local\n"
            "  aegaeon deploy Qwen/Qwen3-4B --replicas 2\n"
            "  aegaeon undeploy Qwen/Qwen3-4B"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", title="commands", metavar="COMMAND"
    )

    doctor_cmd = subparsers.add_parser(
        "doctor",
        help="Check runtime and local deployment prerequisites",
        description="Check runtime dependencies and local deployment prerequisites in the active Python environment.",
        formatter_class=_HelpFormatter,
    )
    doctor_cmd.add_argument(
        "--config", help="YAML config for local model, profile, topology and memory checks",
    )
    doctor_cmd.add_argument(
        "--models", default=None,
        help="Comma-separated startup models, matching start --models; requires --config",
    )
    doctor_cmd.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="Override server.tensor_parallel_size, matching start; requires --config",
    )
    doctor_cmd.add_argument(
        "--strict", action="store_true", help="Return nonzero for warnings as well as failures",
    )
    doctor_cmd.add_argument(
        "--check-cuda", action="store_true",
        help="Also test CUDA tensors, native vLLM, visible GPUs and one-page host registration",
    )

    start_cmd = subparsers.add_parser(
        "start",
        help="Start the Aegaeon API server",
        description=(
            "Start serving models from a YAML configuration. Configure engine "
            "topology and memory budgets in its server section."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  aegaeon start --config aegaeon.yaml --ray-address local\n"
            "  aegaeon start --config aegaeon.yaml --models ''\n\n"
            "Defaults shown above are parser defaults; YAML supplies server resources."
        ),
    )
    start_cmd.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML configuration containing server, models, and devices",
    )
    start_cmd.add_argument(
        "--host", type=str, default="0.0.0.0", help="API server bind address",
    )
    start_cmd.add_argument(
        "--port", type=int, default=8000, help="API server port",
    )
    start_cmd.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated startup models; overrides server.startup_models. "
            "An empty string starts without models; if neither is set, load all models"
        ),
    )

    start_cmd.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help=(
            "Tensor parallel size; only 1 is supported. A nonzero value overrides "
            "server.tensor_parallel_size; omitted or 0 uses YAML"
        ),
    )
    start_cmd.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help=(
            "Ray address, or 'local' to create a runtime; otherwise use "
            "AEGAEON_RAY_ADDRESS, RAY_ADDRESS, then 'auto'"
        ),
    )
    resources = start_cmd.add_argument_group(
        "legacy resource fallbacks",
        "Normal startup uses server YAML fields, including their defaults. "
        "Set these resources in YAML rather than through fallback options.",
    )
    resources.add_argument(
        "--num-prefill-engines", type=int, default=1,
        help="Fallback Prefill engines per node; prefer server.num_prefill_engines",
    )
    resources.add_argument(
        "--num-decode-engines", type=int, default=1,
        help="Fallback Decode engines per node; prefer server.num_decode_engines",
    )
    resources.add_argument(
        "--num-engines", type=int, default=0,
        help="Fallback Simple engines per node; prefer server.num_engines",
    )
    resources.add_argument(
        "--model-cache-size-gb", type=int, default=20,
        help="Fallback CPU weight-cache size in GiB; prefer server.model_cache_size_gb",
    )
    resources.add_argument(
        "--cpu-num-slabs", type=int, default=15,
        help="Fallback CPU KV slab count; prefer server.cpu_num_slabs",
    )
    resources.add_argument(
        "--cpu-slab-size-gb", type=int, default=1,
        help="Fallback CPU KV slab size in GiB; prefer server.cpu_slab_size_bytes",
    )

    deploy_cmd = subparsers.add_parser(
        "deploy",
        help="Deploy a model to an Aegaeon server",
        description="Deploy CPU weight-cache replicas; partial operations exit with status 1.",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  aegaeon deploy Qwen/Qwen3-4B --path /srv/models/Qwen3-4B --replicas 2\n"
            "  aegaeon deploy Qwen/Qwen3-4B --nodes node_0 node_1"
        ),
    )
    deploy_cmd.add_argument("model", type=str, help="Model name (e.g. Qwen/Qwen3-4B)")
    deploy_cmd.add_argument(
        "--path",
        type=str,
        default=None,
        help="Local checkpoint path; if omitted, use the registered path or download",
    )
    deploy_cmd.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Aegaeon API host",
    )
    deploy_cmd.add_argument(
        "--port", type=int, default=8000, help="Aegaeon API port"
    )
    deploy_cmd.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override total context length in tokens; otherwise keep the model/config value",
    )
    deploy_cmd.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Minimum READY replica target for automatic placement",
    )
    deploy_cmd.add_argument(
        "--nodes",
        nargs="+",
        default=None,
        help="Explicit physical node IDs (e.g. node_0 node_1); bypass automatic placement",
    )
    deploy_cmd.add_argument(
        "--tool-parser",
        type=str,
        default=None,
        help="vLLM tool parser override for model aliases",
    )

    undeploy_cmd = subparsers.add_parser(
        "undeploy",
        help="Remove a model from an Aegaeon server",
        description=(
            "Unload selected or all replicas. Retry busy replicas after requests "
            "finish; partial operations exit with status 1."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  aegaeon undeploy Qwen/Qwen3-4B --nodes node_1\n"
            "  aegaeon undeploy Qwen/Qwen3-4B"
        ),
    )
    undeploy_cmd.add_argument("model", type=str, help="Model name (e.g. Qwen/Qwen3-4B)")
    undeploy_cmd.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Aegaeon API host",
    )
    undeploy_cmd.add_argument(
        "--port", type=int, default=8000, help="Aegaeon API port"
    )
    undeploy_cmd.add_argument(
        "--nodes",
        nargs="+",
        default=None,
        help="Only unload replicas on these nodes; omit to unload all replicas",
    )

    args = parser.parse_args()

    if args.command == "start":
        from aegaeon.preflight import ensure_runtime

        try:
            ensure_runtime(config_path=args.config, models=args.models,
                           tensor_parallel_size=args.tensor_parallel_size)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        import uvicorn
        from aegaeon.api import app, lifespan

        args.model_config = args.config
        app.state.engine_args = args
        app.router.lifespan_context = lifespan
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "doctor":
        from aegaeon.preflight import check_runtime

        if args.config is None and (args.models is not None or args.tensor_parallel_size is not None):
            parser.error("doctor --models/--tensor-parallel-size requires --config")
        results = check_runtime(check_cuda=args.check_cuda, config_path=args.config,
                                models=args.models, tensor_parallel_size=args.tensor_parallel_size)
        for result in results:
            print(f"{result.status} {result.name}: {result.detail}")
        failed = any(not result.ok if args.strict else result.blocking for result in results)
        sys.exit(1 if failed else 0)
    elif args.command == "deploy":
        _cmd_deploy(args)
    elif args.command == "undeploy":
        _cmd_undeploy(args)
    else:
        parser.print_help()


def _check_partial_result(result):
    if result.get("status") == "partial" or result.get("failed"):
        print(
            "Operation partially completed; failed nodes: "
            + json.dumps(result.get("failed", {}), ensure_ascii=False),
            file=sys.stderr,
        )
        sys.exit(1)


def _cmd_deploy(args):
    body = {"model": args.model, "replica_count": args.replicas}
    if args.path:
        body["model_path"] = args.path
    if args.max_model_len is not None:
        body["max_model_len"] = args.max_model_len
    if args.nodes:
        body["node_ids"] = args.nodes
    if args.tool_parser:
        body["tool_parser"] = args.tool_parser

    data = json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection(args.host, args.port, timeout=300)

    print(f"Deploying model '{args.model}' at {args.host}:{args.port} ...")
    try:
        conn.request(
            "POST",
            "/v1/models/deploy",
            body=data,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        result = json.loads(resp.read().decode("utf-8"))
        if resp.status == 200:
            _check_partial_result(result)
            ready_nodes = ", ".join(result.get("ready_nodes", []))
            print(
                f"OK Model '{result.get('model', args.model)}' deployed"
                + (f" on {ready_nodes}" if ready_nodes else "")
            )
        else:
            detail = result.get("detail", "")
            if isinstance(detail, dict):
                msg = detail.get("message", resp.reason)
            elif isinstance(detail, str):
                msg = detail
            else:
                msg = resp.reason
            print(f"Error ({resp.status}): {msg}", file=sys.stderr)
            sys.exit(1)
    except ConnectionRefusedError:
        print(f"Error: Connection refused at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def _cmd_undeploy(args):
    body = {"model": args.model}
    if args.nodes:
        body["node_ids"] = args.nodes
    data = json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection(args.host, args.port, timeout=300)

    print(f"Undeploying model '{args.model}' from {args.host}:{args.port} ...")
    try:
        conn.request(
            "POST",
            "/v1/models/undeploy",
            body=data,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        result = json.loads(resp.read().decode("utf-8"))
        if resp.status == 200:
            _check_partial_result(result)
            remaining_nodes = ", ".join(result.get("remaining_nodes", []))
            print(
                f"OK Model '{result.get('model', args.model)}' undeployed"
                + (f"; remaining replicas: {remaining_nodes}" if remaining_nodes else "")
            )
        else:
            detail = result.get("detail", "")
            if isinstance(detail, dict):
                msg = detail.get("message", resp.reason)
            elif isinstance(detail, str):
                msg = detail
            else:
                msg = resp.reason
            print(f"Error ({resp.status}): {msg}", file=sys.stderr)
            sys.exit(1)
    except ConnectionRefusedError:
        print(f"Error: Connection refused at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
