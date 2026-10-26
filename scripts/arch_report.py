#!/usr/bin/env python3
"""Render the arch smoke results as a Markdown compatibility matrix."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=os.path.join(ROOT, "results", "arch_smoke.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "arch_compat_report.md"))
    args = ap.parse_args()

    rows = []
    with open(args.result) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("no results")
        sys.exit(1)

    lines = ["# Aegaeon Architecture Compatibility (Phase 1 smoke)", "", "| Model | Arch | Risk | Endpoint | Tokens | Finish | Status |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        status = r["status"]
        mark = {"pass": "PASS", "timeout": "TIMEOUT"}.get(status, status.upper())
        lines.append(
            f"| {r['name']} | {r.get('arch', '')} | {r.get('risk', '')} | "
            f"{r.get('endpoint', '')} | {r.get('completion_tokens', '-')} | "
            f"{r.get('finish_reason', '')} | {mark} |"
        )
    lines.append("")
    n_pass = sum(1 for r in rows if r["status"] == "pass")
    lines.append(f"**{n_pass}/{len(rows)} pass**" if rows else "")
    lines.append("")
    lines.append(
        "Diagnosis (dummy-weights smoke, 2xA100-40GB, vLLM 0.26.0). Two worker-layer "
        "gaps were fixed in this session: `init_workspace_manager()` in "
        "`aegaeon/worker.py::Worker.switch` (vLLM 0.26 fused-MoE/MLA kernels need the "
        "workspace that vLLM's GPUModelRunner would otherwise initialise), and "
        "`get_model_size_in_bytes()` returning 0 under `AEGAEON_DUMMY_WEIGHTS=1` "
        "in `aegaeon/config.py` (without weights, the scheduler's `_assign_quota` "
        "failed on `n_params()`). These unlocked all but one architecture."
    )
    lines.append("")
    lines.append("- **Gemma2ForCausalLM** (`attn_logit_softcapping`): still rejected by "
                 "`aegaeon/worker.py` (\"Please use Flashinfer backend for models with "
                 "logits_soft_cap (i.e., Gemma-2)\") because aegaeon's metadata path is "
                 "FlashAttention-only. This is a deliberate correctness guard, not a bug.")
    lines.append("")
    lines.append("Phase 2 (real-weights) remains meaningful for all passing architectures; "
                 "large dummy models (DeepSeek-V2-Lite 16B, Qwen3-30B-A3B, Mixtral 8x7B) "
                 "pass the dummy smoke but their real weights exceed the local 40GB GPU "
                 "and were not loaded.")
    lines.append("")

    # Failure details
    fails = [r for r in rows if r["status"] != "pass"]
    if fails:
        lines.append("## Failures", )
        for r in fails:
            lines.append(f"### {r['name']} ({r.get('arch', '')})")
            err = r.get("error") or "(no error; 200 response with zero completion tokens)"
            lines.append(f"```\n{err[:600]}\n```")
        lines.append("")

    out = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
