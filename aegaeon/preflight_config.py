"""Read-only local deployment checks without importing the serving stack."""

import json
import math
import os
from pathlib import Path
import re

from .device_names import canonical_device_name
from .preflight import CheckResult
from .resources import profile_root

GIB = 1024**3
_LENGTH_KEYS = ('max_position_embeddings', 'n_positions', 'max_seq_len',
                'max_sequence_length', 'max_seq_length', 'seq_length', 'seq_len')
_DTYPE_BITS = {
    'BOOL': 8, 'U8': 8, 'I8': 8, 'I16': 16, 'U16': 16, 'I32': 32,
    'U32': 32, 'I64': 64, 'U64': 64, 'F16': 16, 'BF16': 16, 'F32': 32,
    'F64': 64, 'F8_E4M3': 8, 'F8_E5M2': 8, 'F8_E8M0': 8,
    'F8_E4M3FN': 8, 'F8_E4M3FNUZ': 8, 'F8_E5M2FNUZ': 8,
}


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}, found {value!r}')
    return value


def _json(path, limit=32 * 1024**2):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'{path}: JSON exceeds the {limit}-byte preflight limit')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def _relative_key(value):
    value = str(value).strip().replace('\\', '/')
    if not value or Path(value).is_absolute() or '..' in value.split('/'):
        raise ValueError(f'Unsafe or empty profile alias: {value!r}')
    return value


def _model_files(entry):
    # Like ModelSpec.path(), relative paths refer to the invoking working
    # directory. Do not expand ~ or silently resolve against the YAML parent.
    directory = Path(entry['path'])
    if not directory.is_dir():
        raise ValueError(f'Model directory does not exist: {directory}')
    config = _json(directory / 'config.json')
    tokenizer_config = directory / 'tokenizer_config.json'
    tokenizer_metadata = _json(tokenizer_config) if tokenizer_config.exists() else {}
    tokenizer_files = [directory / name for name in
                       ('tokenizer.json', 'tokenizer.model', 'spiece.model', 'vocab.txt')]
    tokenizer_files += ([directory / 'vocab.json', directory / 'merges.txt']
                        if all((directory / name).is_file() for name in ('vocab.json', 'merges.txt')) else [])
    tokens = [path for path in tokenizer_files if path.is_file()]
    if not tokens:
        raise ValueError(f'No local tokenizer assets under {directory}')
    for path in tokens:
        with path.open('rb') as stream:
            if not stream.read(1):
                raise ValueError(f'Empty tokenizer asset: {path}')
    files = sorted(directory.glob('*.safetensors'))
    if not files:
        raise ValueError(f'No .safetensors weights under {directory}')
    index = directory / 'model.safetensors.index.json'
    weight_map = _json(index).get('weight_map') if index.exists() else None
    if index.exists() and (not isinstance(weight_map, dict) or not weight_map):
        raise ValueError(f'{index}: missing or empty weight_map')
    for shard in set(weight_map.values()) if weight_map else ():
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith('.safetensors'):
            raise ValueError(f'{index}: unsafe shard name {shard!r}')
        if not (directory / shard).is_file():
            raise ValueError(f'Missing weight shard: {directory / shard}')
    parameters, storage_bytes = 0, 0
    tensors_by_file = {}
    for path in files:
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Truncated safetensors file: {path}')
            length = int.from_bytes(prefix, 'little')
            if not 1 <= length <= 32 * 1024**2 or length + 8 > path.stat().st_size:
                raise ValueError(f'Invalid safetensors header size: {path}')
            header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError(f'Invalid safetensors header: {path}')
        tensors_by_file[path.name] = set(header) - {'__metadata__'}
        payload_size = path.stat().st_size - 8 - length
        storage_bytes += payload_size
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            if not isinstance(tensor, dict):
                raise ValueError(f'Invalid tensor metadata: {path}:{name}')
            shape, offsets = tensor.get('shape'), tensor.get('data_offsets')
            if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
                raise ValueError(f'Invalid tensor shape: {path}:{name}')
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= payload_size):
                raise ValueError(f'Truncated/invalid tensor offsets: {path}:{name}')
            bits = _DTYPE_BITS.get(tensor.get('dtype'))
            if bits is None or (math.prod(shape) * bits + 7) // 8 != offsets[1] - offsets[0]:
                raise ValueError(f'Unsupported dtype or shape/data size mismatch: {path}:{name}')
            parameters += math.prod(shape)
    if parameters <= 0:
        raise ValueError(f'No tensor parameters in {directory}')
    if weight_map:
        for tensor, shard in weight_map.items():
            if tensor not in tensors_by_file.get(shard, set()):
                raise ValueError(f'{index}: tensor {tensor!r} absent from shard {shard}')
    explicit_params = entry.get('params')
    if explicit_params is not None:
        if (isinstance(explicit_params, bool) or not isinstance(explicit_params, (int, float))
                or not math.isfinite(explicit_params) or explicit_params <= 0):
            raise ValueError('params must be a finite positive number in Gi-parameters')
        parameters = explicit_params * GIB
    if entry.get('max_model_len') is not None:
        max_len = _integer(entry['max_model_len'], 'max_model_len', 1)
    else:
        text_config = config.get('text_config', {})
        if not isinstance(text_config, dict):
            text_config = {}
        lengths = [config.get(key, text_config.get(key)) for key in _LENGTH_KEYS]
        lengths = [value for value in lengths if type(value) is int and value > 0]
        max_len = min(lengths) if lengths else tokenizer_metadata.get('model_max_length', 2048)
        if not isinstance(max_len, (int, float)) or not math.isfinite(max_len) or max_len >= 10**30:
            max_len = 2048
        max_len = _integer(int(max_len), 'model context length', 1)
    return parameters, storage_bytes, max_len, len(files)


def _profile_directory(entry, device, cuda_name):
    configured = entry.get('profile') or ''
    name = str(entry['name'])
    canonical = configured or re.sub(r'[^a-z0-9]+', '_', name.replace('\\', '/').rstrip('/').rsplit('/', 1)[-1].lower()).strip('_')
    model_keys = [_relative_key(canonical)]
    if not configured:
        model_keys.append(_relative_key(name.replace('\\', '/').strip('/')))
    device_keys = []
    if device.get('profile'):
        device_keys.append(_relative_key(device['profile']))
    if cuda_name:
        device_keys.append(_relative_key(cuda_name.removeprefix('NVIDIA ').replace(' ', '_')))
    device_keys.append(_relative_key(device['name'].removeprefix('NVIDIA ').replace(' ', '_')))
    candidates = [profile_root() / model / hardware
                  for model in dict.fromkeys(model_keys) for hardware in dict.fromkeys(device_keys)]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _check_profile(entry, device, cuda_name, max_len):
    directory = _profile_directory(entry, device, cuda_name)
    cases = [(length, batch) for length in (32, 64, 128, 256, 512, 1024, 2048, 4096)
             for batch in (1, 2, 4, 8, 16, *range(30, 210, 10))
             if length * batch <= max_len and (length + 10) * batch < 10240]
    count = 0
    for index, (length, batch) in enumerate(cases):
        # StageEngine and SimpleEngine initialize DecodeEstimator. Its held-out
        # split differs from PrefillEstimator, which is not used at startup.
        if index % 4 == 1:
            continue
        path = directory / f'i{length}b{batch}.json'
        if not path.is_file():
            continue
        data = _json(path)
        values = data.get('decode_latencies')
        if (not isinstance(values, list) or len(values) < 3 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0 for value in values)):
            raise ValueError(f'{path}: decode_latencies needs >=3 finite positive samples in seconds')
        count += 1
    if not count:
        raise ValueError(f'No usable decode training profiles under {directory}; '
                         'check model/device profile aliases and AEGAEON_PROFILE_ROOT')
    return f'{directory}: {count} decode training points; raw samples in seconds, TP=1'


def check_configuration(config_path, *, models=None, tensor_parallel_size=None, cuda_devices=None):
    """Return local asset diagnostics and the per-node shared-memory budget."""
    results = []
    if config_path is None:
        return [CheckResult('Deployment config', False,
                'Supply --config PATH to check topology, selected models, profiles and memory budget.', 'warning')], None
    try:
        import yaml
        with Path(config_path).open() as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError('Config must be a YAML mapping')
        server = raw.get('server', {})
        if not isinstance(server, dict):
            raise ValueError('server must be a mapping')
        nnodes = _integer(server.get('nnodes', 1), 'server.nnodes', 1)
        tp = _integer(tensor_parallel_size or server.get('tensor_parallel_size', 1), 'tensor_parallel_size', 1)
        if tp != 1:
            raise ValueError('Only tensor_parallel_size=1 is supported by this serving runtime')
        simple = _integer(server.get('num_engines', 0), 'server.num_engines')
        prefill = _integer(server.get('num_prefill_engines', 1), 'server.num_prefill_engines')
        decode = _integer(server.get('num_decode_engines', 1), 'server.num_decode_engines')
        if (simple and (prefill or decode)) or (not simple and (prefill < 1 or decode < 1)):
            raise ValueError('Choose simple num_engines>0 with prefill/decode=0, or positive prefill and decode engines')
        engines = simple or prefill + decode
        slabs = _integer(server.get('cpu_num_slabs', 128), 'server.cpu_num_slabs', 1)
        slab_size = _integer(server.get('cpu_slab_size_bytes', GIB), 'server.cpu_slab_size_bytes', 2)
        if slab_size % 2:
            raise ValueError('server.cpu_slab_size_bytes must be divisible by 2')
        cache_gb = _integer(server.get('model_cache_size_gb', 0), 'server.model_cache_size_gb')
        entries = raw.get('models', [])
        if not isinstance(entries, list) or any(not isinstance(item, dict) or not isinstance(item.get('name'), str) or not item['name'].strip() for item in entries):
            raise ValueError('models must be a list of mappings with non-empty names')
        by_name = {entry['name']: entry for entry in entries}
        if len(by_name) != len(entries):
            raise ValueError('Duplicate model names in config')
        ids = [entry.get('id') for entry in entries]
        if any(value is not None for value in ids) and not all(value is not None for value in ids):
            raise ValueError('Mixed model id configuration: set id for every model or omit all ids')
        for value in ids:
            if value is not None:
                _integer(value, 'model id')
        configured = server.get('startup_models')
        if configured is not None and (not isinstance(configured, list) or any(not isinstance(item, str) or not item.strip() for item in configured) or len({item.strip() for item in configured}) != len(configured)):
            raise ValueError('server.startup_models must be a list of unique non-empty names')
        selected = list(dict.fromkeys(name.strip() for name in models.split(',') if name.strip())) if models is not None else ([name.strip() for name in configured] if configured is not None else list(by_name))
        unknown = set(selected) - set(by_name)
        if unknown:
            raise ValueError(f'Startup models absent from config: {sorted(unknown)}')
        devices = raw.get('devices', [])
        if not isinstance(devices, list) or any(not isinstance(item, dict) or not isinstance(item.get('name'), str) or not item['name'].strip() for item in devices):
            raise ValueError('devices must be a list of mappings with non-empty names')
        if len({device['name'] for device in devices}) != len(devices):
            raise ValueError('Duplicate device names in config')
        for device in devices:
            for field in ('memory', 'bandwidth'):
                value = device.get(field)
                if field == 'bandwidth' and value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Device {device["name"]}: {field} must be finite and positive')
        results.append(CheckResult('Deployment config', True,
            f'{config_path}: {nnodes} node(s), {engines} engine(s)/node, TP=1; selected models={selected}'))
    except Exception as exc:
        return [CheckResult('Deployment config', False, f'{type(exc).__name__}: {exc}')], None

    if cuda_devices is not None:
        results.append(CheckResult('Local GPU capacity', len(cuda_devices) >= engines,
            f'{len(cuda_devices)} visible GPU(s), {engines} required per node; CUDA_VISIBLE_DEVICES applies'))
        active_devices = []
        for name in dict.fromkeys(cuda_devices[:engines]):
            alias = canonical_device_name(name)
            device = next((item for item in devices
                           if canonical_device_name(item['name']) == alias), {'name': name})
            active_devices.append((device, name))
    else:
        active_devices = [(device, None) for device in devices]
        results.append(CheckResult('Local GPU capacity', False,
            f'{engines} GPU(s) required per node; use --check-cuda on an allocated worker to inspect visible devices.', 'warning'))
    if nnodes > 1:
        results.append(CheckResult('Worker scope', False,
            f'This check covers only the current host; run doctor with this config on each of the {nnodes} workers. Ray resources are checked during service startup.', 'warning'))

    parameters, assets_complete = 0, True
    payloads = {}
    for name in selected:
        entry = by_name[name]
        if not entry.get('path'):
            assets_complete = False
            results.append(CheckResult(f'Model {name}', False,
                'No local path configured; startup can download this model. Preflight does not download or verify remote weights/tokenizers.', 'warning'))
            continue
        try:
            count, storage, max_len, shards = _model_files(entry)
            parameters += count
            # QuickCache uses the literal configured path as its cache key.
            payloads[str(entry['path'])] = storage
            results.append(CheckResult(f'Model {name}', True,
                f'{entry["path"]}: readable config/tokenizer and {shards} safetensors header(s); index shards checked, weights not loaded'))
        except Exception as exc:
            assets_complete = False
            results.append(CheckResult(f'Model {name}', False, f'{type(exc).__name__}: {exc}'))
            continue
        if not active_devices:
            results.append(CheckResult(f'Profiles {name}', False,
                'No device configured/detected; add devices to YAML or use --check-cuda to resolve the local GPU profile.', 'warning'))
        for device, cuda_name in active_devices:
            try:
                detail = _check_profile(entry, device, cuda_name, max_len)
                if cuda_devices is None:
                    detail += '; static declared-device candidate; actual worker GPU remains unverified'
                results.append(CheckResult(f'Profiles {name}/{device["name"]}', True, detail))
            except Exception as exc:
                detail = f'{type(exc).__name__}: {exc}'
                if cuda_devices is None:
                    detail += '; declared device is not confirmed active. Run doctor --config PATH --check-cuda on the allocated worker to verify its actual profile.'
                results.append(CheckResult(f'Profiles {name}/{device["name"]}', False, detail,
                                           'warning' if cuda_devices is None else 'error'))
    cache = cache_gb * GIB if cache_gb else (int(parameters * 2 * 1.2) if selected and assets_complete else (20 * GIB if not selected else None))
    if cache is None:
        results.append(CheckResult('Shared-memory budget', False,
            f'KV cache={slabs * slab_size} bytes; automatic model cache unresolved until all selected model assets are available.', 'warning'))
        return results, None
    required_payload = sum(payloads.values())
    if cache < required_payload:
        results.append(CheckResult('Model cache capacity', False,
            f'Configured/estimated model cache={cache} bytes is smaller than the selected startup model payload total={required_payload} bytes. Increase server.model_cache_size_gb.'))
    budget = slabs * slab_size + cache
    results.append(CheckResult('Shared-memory budget', True,
        f'Per node: KV={slabs * slab_size} + model={cache} = {budget} bytes; '
        f'{"explicit" if cache_gb else "automatic estimate (2 bytes/parameter, 20% overhead; empty startup defaults to 20 GiB)"}. '
        'Budget excludes Ray and other processes; full-size allocation/registration is not exercised.'))
    return results, budget
