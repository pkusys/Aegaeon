"""Reconstruct verbose workload from actual traces"""

from typing_extensions import Tuple, Optional
from matplotlib import pyplot as plt
from collections import OrderedDict
from pathlib import Path
from . import SIMDATA, ModelType
from .request import Request
from .estimator import *

import numpy as np
import csv
import simpy
import logging

logger = logging.getLogger(__name__)

def gen_workload(
    env: 'simpy.Environment',
    file: str | Path = 'traces-8-13-14.csv',
    seed: Optional[int] = None,
) -> list[Tuple[int, Request]]:
    """Load a CSV path or bundled trace, preserving explicit per-row lengths."""
    rng = np.random.default_rng(seed=seed)

    # Read the arrival trace
    arrivals_by_model = OrderedDict()
    num_reqs = 0
    path = Path(file)
    if not path.is_absolute() and not path.exists():
        path = SIMDATA.joinpath(path)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if len(columns) not in (2, 4):
            raise ValueError(
                f'{path}: expected two trace columns or four columns including '
                'positive integer Input Length and Output Length'
            )
        explicit_lengths = len(columns) == 4
        n = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}:{reader.line_num}: incomplete or extra CSV columns')
            time, model, *io_len = list(row.values())
            hour, minute, second = map(float, time.split(':'))
            ts = int((hour * 3600 + minute * 60 + second) * 1000)
            
            if ts >= 120 * 1000:
                # 2 minute
                break

            pair = None
            if explicit_lengths:
                try:
                    pair = tuple(int(value) for value in io_len)
                    if any(value <= 0 for value in pair):
                        raise ValueError('nonpositive length')
                except ValueError as error:
                    raise ValueError(
                        f'{path}:{reader.line_num}: Input Length and Output Length '
                        'must both be positive integers'
                    ) from error

            if (_model := ModelType.from_str(model)) is not None:
                model = _model
            else:
                model = ModelType.extend_alias(model, [
                    ModelType.qwen2_5_7b.value,
                    ModelType.qwen2_5_7b.value,
                    ModelType.qwen2_5_7b.value,
                    ModelType.qwen2_5_7b.value,
                    ModelType.qwen2_5_7b.value,
                    ModelType.qwen2_5_7b.value,
                    ModelType.yi_9b.value,
                    ModelType.yi_9b.value,
                    ModelType.llama2_13b.value,
                    ModelType.llama2_13b.value,
                ][n % 10])
                n += 1
                assert model is not None
            
            arrivals_by_model.setdefault(model, []).append((ts, pair))
            num_reqs += 1

    # Only legacy two-column traces sample their lengths from a separate file.
    io_lengths = []
    if not explicit_lengths and num_reqs:
        with open(SIMDATA.joinpath("io_len.csv"), 'r') as f:
            reader = csv.DictReader(f, fieldnames=["input", "output"])
            for row in reader:
                i, o = int(row["input"]), int(row["output"])
                if i > 0 and o > 0:
                    io_lengths.append((i, o))
        if not io_lengths:
            raise ValueError('No positive input/output lengths in io_len.csv')
    
    # Synthesize the requests
    reqs = []
    id = 0
    sample = rng.choice(len(io_lengths), size=num_reqs) if io_lengths else None
    for model, arrivals in arrivals_by_model.items():
        max_seq_len = ModelType.max_seq_len(model)

        for arrival, pair in arrivals:
            input_len, output_len = pair if pair is not None else io_lengths[sample[id]]

            # Preserve the legacy scaling of lengths that exceed model context.
            if input_len + output_len > max_seq_len:
                scale = max_seq_len / max(input_len + output_len + 32, 16384)
                input_len, output_len = int(input_len * scale), max(1, int(output_len * scale))
            reqs.append((arrival,
                         Request(env=env, req_id=id, model=model,
                                 input_len=input_len, 
                                 output_len=output_len)))
            id += 1
    reqs.sort(key=lambda x:x[0])

    return reqs

def syn_workload(
    env: 'simpy.Environment',
    seed: Optional[int] = None,
    model_num: int = 8,
    arrival_rate: float = 0.25,
    duration_s: int = 3600,
    inlen_scale: float = 1.0,
    outlen_scale: float = 1.0,
    outlen_clip: int = 32,
) -> list[Tuple[int, Request]]:
    """Synthesize workload (list of Requests) from the given parameters."""
    rng = np.random.default_rng(seed=seed)
    reqs = []

    # Read the input/output lengths
    io_lengths = []
    with open(SIMDATA.joinpath("io_len.csv"), 'r') as f:
        reader = csv.DictReader(f, fieldnames=["input", "output"])
        for row in reader:
            i, o = int(row["input"]), int(row["output"])
            if i > 0 and o > 0:
                io_lengths.append((i, o))

    # Prepare models
    models = []
    while len(models) < model_num:
        i = len(ModelType)
        model = ModelType.extend_alias(f'synth-model-{i}', [
            ModelType.yi_9b.value, ModelType.yi_9b.value, ModelType.yi_9b.value, ModelType.llama2_13b.value,
        ][i % 4])
        models.append(model)

    for m in range(model_num):
        time_ms = 0
        model_arrival_rate = arrival_rate / model_num
        num_reqs = int(duration_s * model_arrival_rate)
        intervals_ms = rng.exponential(scale=1/model_arrival_rate, size=num_reqs) * 1000
        io_sample = rng.choice(np.arange(0, len(io_lengths), 1), size=num_reqs)
        for i, interval_ms in enumerate(intervals_ms):
            time_ms += interval_ms 
            model = models[m]
            input_len, output_len = io_lengths[io_sample[i]]
            max_seq_len = ModelType.max_seq_len(model)

            input_len = int(input_len * inlen_scale) 
            output_len = int(output_len * outlen_scale)
            if output_len < outlen_clip: output_len = outlen_clip
            # Scale down the requests that exceed the max-token limit
            if input_len + output_len > max_seq_len:
                # scale = max_seq_len / max(input_len + output_len + 32, 16384)
                scale = max_seq_len / (input_len + output_len + 32)
                input_len, output_len = int(input_len * scale), max(1, int(output_len * scale))
            
            reqs.append((int(time_ms),
                        Request(env=env, req_id=id, model=model,
                                input_len=input_len, 
                                output_len=output_len)))
    
    reqs.sort(key=lambda req: req[0])
    
    print(f'avg inlen: {sum(req.input_len for _, req in reqs) / len(reqs)}')
    print(f'avg outlen: {sum(req.output_len for _, req in reqs) / len(reqs)}')
    print(f'size: {len(reqs)}')
    return reqs


def plot_workload(
    workload: list[Tuple[int, Request]],
    outfile: str,
    C: float = 1.1,
    device: DeviceType = DeviceType.a10,
):
    spans = {}
    ragnarok = 0

    for start, req in workload:
        start = start / 1000
        est1 = make_estimator(PrefillEstimator, model=req.model, device=device)
        est2 = make_estimator(DecodeEstimator, model=req.model, device=device)
        elapsed = est1.predict(req.input_len, 1) \
            + sum(est2.predict(l + req.input_len, 1) for l in range(1, req.output_len+1))
        elapsed = C * elapsed
        end = start + elapsed
        
        if req.model not in spans: spans[req.model] = []
        spans[req.model].append((int(start), int(end)+1))
        ragnarok = max(ragnarok, int(end)+1)

    reqs_model = {model: [0]*(ragnarok+1) for model in spans}
    reqs_tot = [0]*(ragnarok+1)
    models = []
    
    for model, span in spans.items():
        for start, end in span:
            for t in range(start, end):
                reqs_model[model][t] += 1
                reqs_tot[t] += 1
    for t in range(ragnarok+1):
        models.append(sum(1 if reqs[t] > 0 else 0 for reqs in reqs_model.values()))
    
    X = list(range(ragnarok+1))
    plt.figure(figsize=(10, 10))

    plt.subplot(211)
    plt.plot(X, reqs_tot, label="total")
    i = 0
    for model, reqs in reqs_model.items():
        plt.plot(X, reqs, label=str(model))
        i += 1
        if i >= 8: break
    plt.title("Number of Requests over time")
    plt.xlabel("Time")
    plt.ylabel("#req")
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(X, models)
    plt.title("Number of Models over time")
    plt.xlabel("Time")
    plt.ylabel("#model")
    plt.grid(True)

    plt.savefig(f"sim/plots/{outfile}.png")

def save_workload(
    workload: list[Tuple[int, Request]],
    outfile: str,
    full: bool = False,
):
    with open(outfile, 'w') as outfile:
        writer = csv.writer(outfile)
        # Write header
        header = ['Request Time', 'Model Name']
        if full: 
            header.extend(['Input Length', 'Output Length'])
        writer.writerow(header)
        for t, req in workload:
            ms = t % 1000
            s = (t // 1000) % 60
            m = (t // 60000) % 60
            h = t // 3600000
            timestamp = f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'
            row = [timestamp, str(req.model)[10:]]
            if full:
                row.extend([str(req.input_len), str(req.output_len)])
            writer.writerow(row)

if __name__ == "__main__":

    def gen(model_num: int):
        arrival_rate = 0.1 * model_num
        inlen_scale = 0.5
        outlen_scale = 1.0
        outlen_clip = 16
        workload = syn_workload(
            None, seed=0, model_num=model_num, 
            arrival_rate=arrival_rate, duration_s=120, 
            inlen_scale=inlen_scale,
            outlen_scale=outlen_scale,
            outlen_clip=outlen_clip,)

        name = f'synth-{model_num}-{arrival_rate:.1f}-1.0'
        save_workload(workload, f'sim/simdata/new/{name}.csv', full=True)

    for model_num in range(16, 81, 2):
        gen(model_num)
    
