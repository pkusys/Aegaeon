"""Estimate latencies by fitting models"""

from . import SIMDATA, ModelType, DeviceType
from sklearn.linear_model import LinearRegression
from typing_extensions import Tuple, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)
DEFAULT_PROFILE_ROOT = SIMDATA.parent.parent.joinpath("profiles")
PROFILE_ROOT = DEFAULT_PROFILE_ROOT

# Simulator names predate the serving profile aliases. Keep this mapping
# explicit: another checkpoint or an A100 with different capacity is not a
# substitute for a missing model/device profile.
MODEL_PROFILE_ALIASES = {
    "qwen2.5-7b": "qwen2_5_7b",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "yi-9b": "yi1_5_9b_chat",
    "llama2-13b": "llama2_13b_chat",
}
DEVICE_PROFILE_ALIASES = {
    "a10": ("A10",),
    "l20": ("L20",),
    "a100": ("A100",),
    "a100_40gb": ("A100-PCIE-40GB", "A100_40GB"),
    "h800": ("H800",),
    "h100": ("H100",),
}


def set_profile_root(path: Path):
    """Select measured execution profiles and discard cached estimators."""
    global PROFILE_ROOT
    PROFILE_ROOT = Path(path)
    _estimators.clear()


def _profile_path(model: ModelType, device: DeviceType) -> Path:
    model_key, device_key = model.as_str(), str(device)
    # Accept either simulator layout or repository serving layout within the
    # selected root. An explicit custom root remains authoritative.
    model_keys = tuple(dict.fromkeys((
        model_key, MODEL_PROFILE_ALIASES[model_key],
    )))
    device_keys = (device_key, *DEVICE_PROFILE_ALIASES[device_key])
    candidates = [PROFILE_ROOT / name / hardware
                  for name in model_keys for hardware in device_keys]
    for path in candidates:
        if path.is_dir():
            logger.info("Using execution profiles for %s/%s: %s", model_key, device_key, path)
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise ValueError(
        f"No execution profiles for {model_key}/{device_key}. Searched: {searched}. "
        "Expected i<input_length>b<batch_size>.json files with "
        "avg_prefill_latency and avg_decode_latency in seconds. Supply matching "
        "profiles with --profile-root; see sim/README.md."
    )


def validate_profiles(models, devices):
    """Fit both estimators before starting any simulation workers."""
    models = {ModelType.from_str(model.as_str()) for model in models}
    for model in sorted(models, key=lambda model: model.as_str()):
        for device in devices:
            make_estimator(PrefillEstimator, model, DeviceType(device))
            make_estimator(DecodeEstimator, model, DeviceType(device))

def _load_time_profile(
    dir: Path,
    input_len: int,
    batch_size: int,
) -> dict:
    if not (file := dir.joinpath(f"i{input_len}b{batch_size}.json")).exists():
        # logger.warning(f"No profiled data for {file}")
        return None
    with open(file, 'r') as f:
        data = json.load(f)
    return data

class PrefillEstimator:
    """Estimates the latency of the prefill phase for the given model
    
    y = C1 * (4th^2 + 2thm) + C2 * 3hs / b + C3
    t: sum of input lengths (number of tokens in the batch)
    s: squared sum of input lengths
    h: model hidden size
    m: model FFN intermediate size
    b: block size in the FlashAttention kernel

    Since all but the `t` and `s` parameters are constant within a
    set of profiled results, we essentially have:
    y = a + b * t + c * s
    """
    _all_set: list[Tuple[int, int]] 
    _testing_set: list[Tuple[int, int]] 
    _training_set: list[Tuple[int, int]] 

    def __init__(
        self,
        model: ModelType,
        device: DeviceType = DeviceType.a10,
        training_set: list[Tuple[int, int]] = None,
    ):
        path = _profile_path(model, device)
        self.path = path
        self._all_set = [(ilen, bs) 
            for ilen in [16,32,64,128,256,512,1024,2048,4096] 
            for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10))) 
            if ilen*bs <= ModelType.max_seq_len(model) and (ilen+10)*bs < 10240]
        self._testing_set = [sample for i, sample in enumerate(self._all_set) if i % 3 == 0] 
        self._training_set = [sample for i, sample in enumerate(self._all_set) if i % 3 != 0] 
        
        X = []
        y = []
        training_set = training_set if training_set is not None else self._training_set
        for input_len, batch_size in training_set:
            t = input_len * batch_size
            s = input_len**2 * batch_size
            if (data := _load_time_profile(path, input_len, batch_size)) is None:
                continue
            X.append([t, s])
            y.append(data["avg_prefill_latency"])

        if not X:
            raise ValueError(
                f"No usable prefill training profiles under {path}; "
                "see sim/README.md for the expected samples and fields."
            )
        self.model = LinearRegression().fit(X, y)
    
    def predict(
        self,
        input_len: int,
        batch_size: int,
        prefill_len_list: Optional[list[int]] = None,
    ) -> float:
        if prefill_len_list is None:
            t = input_len * batch_size
            s = input_len**2 * batch_size
        else:
            t = sum(prefill_len_list)
            s = sum(l**2 for l in prefill_len_list)

        return self._predict(t, s)
    
    def _predict(self, t, s) -> float:
        return self.model.predict([[t, s]])[0]
    
    def test(
        self,
        testing_set: list[Tuple[int, int]] = None,
    ) -> float:
        X = []
        y = []
        testing_set = testing_set if testing_set is not None else self._testing_set
        for input_len, batch_size in testing_set:
            t = input_len * batch_size
            s = input_len**2 * batch_size
            if (data := _load_time_profile(self.path, input_len, batch_size)) is None:
                continue
            X.append([t, s])
            y.append(data["avg_prefill_latency"])

        return self.model.score(X, y)

class DecodeEstimator:
    """Estimates the latency of the decode phase for the given model
    
    y = C4 * (4h^2 + 2hm) + C5 * 3ht 
    t: sum of input lengths (number of tokens in the batch)
    h: model hidden size
    m: model FFN intermediate size

    Since all but the `t` parameters are constant within a
    set of profiled results, we essentially have:
    y = a + b * t

    As the decode phase may experience a transition from being memory-bound
    to compute-bound as the batch size increases, we append a batch size term
    to account for that:

    y = a + b * t + c * B

    where `B` is the batch size.
    """
    _all_set: list[Tuple[int, int]] 
    _testing_set: list[Tuple[int, int]] 
    _training_set: list[Tuple[int, int]] 

    def __init__(
        self,
        model: ModelType,
        device: DeviceType = DeviceType.a10,
        training_set: list[Tuple[int, int]] = None,
    ):
        path = _profile_path(model, device)
        self.path = path
        self._all_set = [(ilen, bs) 
            for ilen in [16,32,64,128,256,512,1024,2048,4096] 
            for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10))) 
            if ilen*bs <= ModelType.max_seq_len(model) and (ilen+10)*bs < 10240]
        self._testing_set = [sample for i, sample in enumerate(self._all_set) if i % 3 == 0] 
        self._training_set = [sample for i, sample in enumerate(self._all_set) if i % 3 != 0]
        
        X = []
        y = []
        training_set = training_set if training_set is not None else self._training_set
        for input_len, batch_size in training_set:
            t = input_len * batch_size
            b = batch_size
            if (data := _load_time_profile(path, input_len, batch_size)) is None:
                continue
            X.append([t, b])
            y.append(data["avg_decode_latency"])

        if not X:
            raise ValueError(
                f"No usable decode training profiles under {path}; "
                "see sim/README.md for the expected samples and fields."
            )
        self.model = LinearRegression().fit(X, y)
    
    def predict(
        self,
        input_len: int,
        batch_size: int,
        decode_len_list: Optional[list[int]] = None,
    ) -> float:
        if decode_len_list is None:
            t = input_len * batch_size
            b = batch_size
        else:
            t = sum(decode_len_list)
            b = len(decode_len_list)

        return self._predict(t, b)
    
    def _predict(self, t, b) -> float:
        return self.model.predict([[t, b]])[0]
    
    def test(
        self,
        testing_set: list[Tuple[int, int]] = None,
    ) -> float:
        X = []
        y = []
        testing_set = testing_set if testing_set is not None else self._testing_set
        for input_len, batch_size in testing_set:
            t = input_len * batch_size
            b = batch_size
            if (data := _load_time_profile(self.path, input_len, batch_size)) is None:
                continue
            X.append([t, b])
            y.append(data["avg_decode_latency"])

        return self.model.score(X, y)
    

def test_prefill_estimator(model: ModelType = ModelType.qwen2_5_7b, device: DeviceType = DeviceType.h100):
    est = PrefillEstimator(model, device)
    print(f"Score for {model.as_str()}, {device} (prefill): {est.test()}")

def test_decode_estimator(model: ModelType = ModelType.qwen2_5_7b, device: DeviceType = DeviceType.h100):
    est = DecodeEstimator(model, device)
    print(f"Score for {model.as_str()}, {device} (decode): {est.test()}")

_estimators = {}
def make_estimator(
    cls,
    model: ModelType = ModelType.qwen2_5_7b,
    device: DeviceType = DeviceType.a10,
):
    if (est := _estimators.get((model, device, cls))) is not None:
        return est
    est = cls(model, device)
    _estimators[(model, device, cls)] = est
    return est

if __name__ == "__main__":
    test_prefill_estimator(model=ModelType.yi_9b, device=DeviceType.h100)
    test_decode_estimator(model=ModelType.yi_9b, device=DeviceType.h100)
    # print(PrefillEstimator(ModelType.qwen2_5_7b, DeviceType.h100).predict(-1, -1, prefill_len_list=[770, 771, 772]))
