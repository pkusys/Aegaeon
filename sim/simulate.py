from . import DeviceType
from .util import pcie
from .worker import Worker
from .scheduler import *
from .request import Request
from .workload import gen_workload
from .estimator import PROFILE_ROOT, set_profile_root, validate_profiles
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import simpy
import argparse
import json
import logging
import math
import numpy as np
import time
import os
import copy
import traceback

TTFT_SLO = 10000
TPOT_SLO = 100
DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return seconds


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # -- general
    parser.add_argument('--name', type=str, default=None, help="Name of experiment")
    parser.add_argument('--max-batch-size', type=positive_int, default=10 ** 7,
                        help="Maximum batch size")
    parser.add_argument('--max-tokens', type=int, default=10 ** 7,
                        help="Maximum number of token in a batch")
    parser.add_argument('--num-models', type=int, nargs="+")
    parser.add_argument('--arrival-rate', type=float, nargs="+")
    parser.add_argument('--inlen-scale', type=float, default=1.0)
    parser.add_argument('--outlen-scale', type=float, default=1.0)
    # -- workload
    parser.add_argument('--workload', type=Path,
                        help="CSV path to simulate directly instead of a bundled workload sweep")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--profile-root', type=Path, default=PROFILE_ROOT,
                        help="Execution profile root in simulator or repository profiles layout; "
                        "defaults to the repository profiles directory")
    parser.add_argument('--job-timeout', type=positive_seconds, default=300.0,
                        help="Maximum wall-clock seconds per simulation subprocess (default: 300)")
    parser.add_argument('--output-dir', type=Path, default=Path(DIR).parent / 'plots' / 'json',
                        help="Directory for per-token simulation results")
    # -- devices
    parser.add_argument('--devices', nargs='+', type=str, default=['h100'] * 16,
                        help="List of GPU specifications. For example, `--devices a100 a100` "
                        "specifies two NVIDIA A100 gpus for simulation. "
                        "Checkout `DeviceType` for valid GPU types. ")
    parser.add_argument('--h2d', type=pcie, default='gen4x16',
                        help="Host to device PCIe specification. "
                        "Either use the `genGxN` format, or specify the bandwidth "
                        "in GB/s directly.")
    parser.add_argument('--swap-overhead', type=int, default=0,
                        help="Extra model swapping overhead in ms")

    args = parser.parse_args(args=args)
    if args.workload is not None:
        if args.num_models is not None or args.arrival_rate is not None:
            parser.error('--workload cannot be combined with --num-models or --arrival-rate')
    elif args.num_models is None or args.arrival_rate is None:
        parser.error('sweep mode requires both --num-models and --arrival-rate; or supply --workload')

    return args

def main(
    args,
    policy: Policy,
    silent: bool = False,
    ret = None,
) -> list[Request]:
    args.devices = [DeviceType(device) for device in args.devices]
    set_profile_root(getattr(args, 'profile_root', PROFILE_ROOT))

    if not silent:
        print('====================')
        print(args)
        print(policy)

    env = simpy.Environment()

    # Prepare simulation workload
    if args.workload is not None:
        workload = gen_workload(env, file=args.workload, seed=args.seed)
    else:
        workload = gen_workload(env, seed=args.seed)
    validate_profiles((req.model for _, req in workload), args.devices)
    if not silent:
        print('size of workload:', len(workload))

    # Prepare the intra-node workers and scheduler
    scheduler = Scheduler(env=env, policy=policy,
                          workers=None, h2d=args.h2d)
    if isinstance(policy, MixServePolicy):
        assert policy.num_prefill_workers < len(args.devices)
        workers = []
        for wid, device in enumerate(args.devices):
            if wid < policy.num_prefill_workers:
                worker = PrefillWorker(scheduler, env=env, worker_id=wid, device=device, swap_overhead_ms=args.swap_overhead)
            else:
                worker = DecodeWorker(
                    scheduler, 
                    env=env, worker_id=wid, device=device, 
                    swap_overhead_ms=args.swap_overhead,
                    tpot_slo_ms=TPOT_SLO)
            workers.append(worker)
    else:
        workers = [Worker(env=env, worker_id=wid, device=device,
                          swap_overhead_ms=args.swap_overhead,
                          max_batch_size=args.max_batch_size)
                   for wid, device in enumerate(args.devices)]
    scheduler.workers = workers

    # Register processes
    for worker in workers:
        env.process(worker.run())
    scheduler.put_workload(workload)
    
    # Start the simulation
    try:
        start = time.perf_counter()
        env.run()

        end = time.perf_counter()

        if not silent:
            print(f"Simulation successful for {args}\n{policy}\nelapsed = {end-start}s")
        
            if isinstance(policy, MixServePolicy):
                for worker in workers:
                    if isinstance(worker, PrefillWorker):
                        print(f'Worker #{worker.worker_id} max group size: {worker.max_group_sz}; dist = {np.histogram(worker.group_sz, bins=range(1,worker.max_group_sz+2))}')

        if ret is not None:
            ret.put([req for _, req in workload])
        else:
            return [req for _, req in workload]

    except Exception:
        print("Simulation failed")
        # Any exception triggers an information dump
        print(args.name)
        print(policy)
        for worker in workers:
            worker.dump_state()
        raise


def _run_simulation(args, policy, result_queue):
    try:
        result_queue.put(('ok', main(args, policy, silent=True)))
    except Exception:
        result_queue.put(('error', traceback.format_exc()))
        raise


def _run_jobs(jobs, job_timeout):
    """Collect results without waiting forever for a crashed or stuck worker."""
    processes = []
    results = {}
    try:
        for index, (x, policy, args) in enumerate(jobs):
            result_queue = Queue()
            process = Process(target=_run_simulation, args=(args, policy, result_queue))
            processes.append((index, x, policy, args, process, result_queue,
                              time.monotonic() + job_timeout))
            process.start()
        pending = list(processes)
        while pending:
            received = False
            for entry in list(pending):
                index, x, policy, args, process, result_queue, deadline = entry
                label = f'{args.workload} ({policy})'
                try:
                    status, payload = result_queue.get_nowait()
                except Empty:
                    if process.exitcode is not None:
                        raise RuntimeError(
                            f'Simulation {label} exited with code {process.exitcode} without a result'
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'Simulation {label} exceeded {job_timeout:g} seconds')
                    continue
                if status != 'ok':
                    raise RuntimeError(f'Simulation {label} failed:\n{payload}')
                process.join(timeout=max(0, deadline - time.monotonic()))
                if process.is_alive():
                    raise TimeoutError(f'Simulation {label} did not exit within {job_timeout:g} seconds')
                if process.exitcode != 0:
                    raise RuntimeError(f'Simulation {label} exited with code {process.exitcode}')
                results[index] = payload
                pending.remove(entry)
                received = True
            if pending and not received:
                time.sleep(0.05)
        return [results[index] for index in range(len(jobs))]
    finally:
        # Only reap the processes created by this invocation, including on Ctrl-C.
        for _, _, _, _, process, result_queue, _ in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            result_queue.close()
            result_queue.join_thread()


def plot_slo_models(args):
    POLICIES = [
        SeLLMPolicy(),
        SeLLMPlusPolicy(),
    ]
    NMODELS = args.num_models
    RATES = args.arrival_rate
    inlen_scale = args.inlen_scale
    outlen_scale = args.outlen_scale

    ttft_by_x = {}
    tpot_by_x = {}
    reqs_by_x = {}
    
    # Validate every workload/profile before launching subprocesses.
    set_profile_root(getattr(args, 'profile_root', PROFILE_ROOT))
    jobs = []
    for policy in POLICIES:
        if getattr(args, 'workload', None) is not None:
            cases = [(Path(args.workload).stem, args.workload)]
        else:
            cases = [
                (m if len(NMODELS) > 1 else rate,
                 f'synth-{m}-{rate}-{inlen_scale}-{outlen_scale}.csv')
                for m in NMODELS for rate in RATES
            ]
        for x, workload_path in cases:
            job_args = copy.deepcopy(args)
            job_args.workload = workload_path
            workload = gen_workload(simpy.Environment(), file=workload_path, seed=args.seed)
            validate_profiles((req.model for _, req in workload), args.devices)
            jobs.append((x, policy, job_args))
        
    if args.name is not None: print(args.name)
    print(f"Starting {len(jobs)} jobs...")
    start = time.perf_counter()
    results = _run_jobs(jobs, getattr(args, 'job_timeout', 300.0))
    for i, ((x, policy, _), reqs) in enumerate(zip(jobs, results)):
        print(f'({i}/{len(jobs)}) {x} {policy} done after {time.perf_counter()-start:.2f} seconds')

        ttft_by_x[(x, policy)] = []
        tpot_by_x[(x, policy)] = []
        reqs_by_x[(x, policy)] = reqs
        
        for req in reqs:
            if not req.is_finished:
                if not req.is_failed:
                    logger.warning(f'{req} is neither finished nor failed; the scheduler must have missed it')
                ttft_by_x[(x, policy)].append(10 ** 7) 
                tpot_by_x[(x, policy)].append(10 ** 7) 
                continue
            ttft_by_x[(x, policy)].append(req.time_first_token_ms - req.time_arrival_ms)
            tpot_by_x[(x, policy)].append(
                int((req.time_last_token_ms - req.time_first_token_ms) / (req.output_len - 1)) \
                if req.output_len > 1 \
                else 0
            )
    
    json_name = args.name if args.name else "slo-models"
    output_dir = Path(getattr(args, 'output_dir', Path(DIR).parent / 'plots' / 'json'))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f'{json_name}.json'
    with open(json_path, 'w') as f:
        metrics = {}
        for k, reqs in reqs_by_x.items():
            m = {}
            for req in reqs:
                if req.is_failed:
                    m[req.req_id] = [-1]
                    continue
                per_token_ms = [req.time_first_token_ms - req.time_arrival_ms]
                for end, start in zip(req.time_each_token_ms[1:], req.time_each_token_ms[:-1]):
                    per_token_ms.append(end-start)
                m[req.req_id] = per_token_ms
            # Policy objects have process-specific repr addresses; persist names.
            x, policy = k
            metrics[str((x, str(policy)))] = m
            
        json.dump({str(k): v for k, v in metrics.items()}, f)



if __name__ == '__main__':
    args = parse_args()

    # Set logging level
    logging.basicConfig(level=logging.WARNING)

    if args.name is None:
        if args.workload is not None:
            args.name = f'sim-{args.workload.stem}'
        else:
            model_spec = f'{args.num_models[0]}' if len(args.num_models) == 1 else f'{args.num_models[0]}-{args.num_models[-1]}'
            rate_spec = f'{args.arrival_rate[0]}' if len(args.arrival_rate) == 1 else f'{args.arrival_rate[0]}-{args.arrival_rate[-1]}'
            args.name = f"sim-model{model_spec}-rate{rate_spec}-scale-{args.inlen_scale}-{args.outlen_scale}"
    try:
        plot_slo_models(args)
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from exc
