"""Reading workload files"""

import argparse
import os

from typing import Tuple, Optional, List

from aegaeon.logger import init_logger
from aegaeon.models import ModelType
from aegaeon.model_registry import ModelSpec
from aegaeon.config import get_model_config
from aegaeon.utils import get_tokenizer

import json
import csv
import numpy as np
from functools import cache

logger = init_logger(__name__)

sharegpt = os.environ.get("AEGAEON_SHAREGPT_PATH", "/srv/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
_dataset = None

@cache
def get_dataset() -> list[Tuple[str, str]]:
    global _dataset
    if _dataset:
        return _dataset 

    dataset = []
    with open(sharegpt, "r") as f:
        convs = json.load(f)
    for conv in convs:
        i, o = "", None
        for req in conv["conversations"]:
            if req["from"] == "human":
                i += req["value"]
            elif req["from"] == "gpt":
                o = req["value"]
                if len(i) == 0:
                    continue
                dataset.append((i, o))
                i += req["value"]

    _dataset = dataset
    return _dataset


def syn_workload(
    seed: Optional[int] = None,
    model_num: int = 32,
    arrival_rate: float = 0.1,
    inlen_scale: float = 1.0,
    outlen_scale: float = 1.0,
    duration: int = 120,
) -> list[Tuple[ModelSpec, float, List[int], int]]:
    """Synthesize workload (list of Requests) from the given parameters."""
    if seed is None:
        seed = 0
    rng = np.random.default_rng(seed=seed)
    reqs = []

    # Read the dataset
    dataset = get_dataset()

    # Synthesize
    ilens = []
    olens = []
    from aegaeon.model_registry import get_registry
    all_models = list(get_registry())
    for i in range(model_num):
        model = all_models[i % len(all_models)]
        accum = 0
        num_reqs = int(duration * arrival_rate)
        intervals = rng.exponential(scale=1 / arrival_rate, size=num_reqs)
        req_sample = rng.choice(np.arange(0, len(dataset), 1), size=num_reqs)
        for i, interval in enumerate(intervals):
            accum += interval
            if accum > duration:
                break

            intext, outtext = dataset[req_sample[i]]

            # make sure controller gets the chance to download models
            tokenizer = get_tokenizer(model.model_path or model.name)
            model_config = get_model_config(model)
            max_model_len = model_config.max_model_len
            if max_model_len < 2:
                raise ValueError(
                    f"Invalid max_model_len={max_model_len} for {model.name}: "
                    "a workload request needs at least one input and one output token"
                )

            prompt_token_ids = tokenizer.encode(intext)
            decode_tokens = len(tokenizer.encode(outtext))
            ilen = max(1, int(len(prompt_token_ids) * inlen_scale))
            minimum_output_len = min(16, max_model_len - 1)
            olen = max(minimum_output_len, int(decode_tokens * outlen_scale))

            # Scale seqs that are too long or too short
            if ilen + olen > max_model_len:
                scale = max_model_len / (ilen + olen + 32)
                ilen = max(1, int(ilen * scale))
                olen = max(minimum_output_len, int(olen * scale))
                olen = min(olen, max_model_len - 1)
                ilen = min(ilen, max_model_len - olen)

            ilens.append(ilen)
            olens.append(olen)

            reqs.append((model, accum, [1] * ilen, olen))

    logger.info(
        f"Workload({model_num}, {arrival_rate}, "
        f"size={len(reqs)}, "
        f"avg-ilen={sum(ilens)/len(ilens):.1f}, "
        f"avg-olen={sum(olens)/len(olens):.1f})"
    )
    reqs.sort(key=lambda req: req[1])
    return reqs


def save_workload(
    workload: List[Tuple[ModelSpec, float, List[int], int]],
    outfile: str,
):
    with open(outfile, "w") as outfile:
        writer = csv.writer(outfile)
        # Write header
        header = ["Request Time", "Model Name", "Input Length", "Output Length"]

        writer.writerow(header)
        for model, t, input_ids, output_len in workload:
            t_ms = int(t * 1000)
            ms = t_ms % 1000
            s = (t_ms // 1000) % 60
            m = (t_ms // 60000) % 60
            h = t_ms // 3600000
            timestamp = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
            row = [timestamp, str(model), str(len(input_ids)), str(output_len)]
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-models', type=int, nargs='+')
    parser.add_argument('--arrival-rate', type=float, nargs='+')
    parser.add_argument('--inlen-scale', type=float, default=1.0)
    parser.add_argument('--outlen-scale', type=float, default=1.0)
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()

    for m in args.num_models:
        for r in args.arrival_rate:
            name = f"synth-{m}-{r}-{args.inlen_scale:.1f}-{args.outlen_scale:.1f}"
        
            save_workload(
                syn_workload(model_num=m, arrival_rate=r, 
                             inlen_scale=args.inlen_scale, outlen_scale=args.outlen_scale, 
                             duration=args.duration),
                f"workloads/{name}.csv",
            )
