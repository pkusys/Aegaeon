"""Estimate latencies by fitting models"""

from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, List
from pathlib import Path
import json

from aegaeon.models import ModelType
from aegaeon.model_registry import ModelSpec
from aegaeon.device_registry import DeviceSpec
from aegaeon.config import get_model_config
from aegaeon.utils import DeviceType
from aegaeon.resources import profile_root


def _resolve_default_device():
    from aegaeon.device_registry import get_device_registry as get_dev_reg

    try:
        import torch

        name = torch.cuda.get_device_name(0)
        spec = get_dev_reg().from_str(name)
        if spec is not None:
            return spec
    except Exception:
        pass
    return next(iter(get_dev_reg()))


def _profile_dir(model: ModelSpec, device: DeviceSpec) -> Path:
    root = profile_root()
    model_keys=[model.profile_key()]
    if not model.profile:
        legacy_key = model.name.replace("\\", "/").strip("/")
        if (
            legacy_key
            and legacy_key != model_keys[0]
            and ".." not in legacy_key.split("/")
        ):
            model_keys.append(legacy_key)
    device_keys = []
    configured_device_key = device.extra.get("profile")
    if configured_device_key:
        device_keys.append(str(configured_device_key).strip())

    # Existing profile data uses the CUDA product name (minus the NVIDIA
    # prefix), while DeviceSpec may use a logical alias such as A100_40GB.
    try:
        import torch

        cuda_name = torch.cuda.get_device_name(0)
        device_keys.append(cuda_name.removeprefix("NVIDIA ").replace(" ", "_"))
    except Exception:
        pass
    device_keys.append(device.name.removeprefix("NVIDIA ").replace(" ", "_"))

    # Preserve order while removing duplicate/empty candidates.
    model_keys = list(dict.fromkeys(key for key in model_keys if key))
    device_keys = list(dict.fromkeys(key for key in device_keys if key))
    candidates = [
        root.joinpath(*model_key.split("/")).joinpath(device_key)
        for model_key in model_keys
        for device_key in device_keys
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Return the canonical location so the missing-profile error remains
    # actionable even when none of the compatibility candidates exist.
    return candidates[0]
    


def _raise_missing_profiles(model: ModelSpec, device: DeviceSpec) -> None:
    directory = _profile_dir(model, device)
    raise FileNotFoundError(
        "No usable latency profiles were found for "
        f"model '{model.name}' (profile alias '{model.profile_key()}') on "
        f"device '{device}'. Expected files such as i32b1.json under "
        f"'{directory}'. Set AEGAEON_PROFILE_ROOT to the directory containing "
        "your measured model/device profiles when installed outside a source checkout."
    )


def _load_time_profile(
    model: ModelSpec,
    input_len: int,
    batch_size: int,
    device: DeviceSpec,
    tp: int = 1,
) -> dict:
    assert tp == 1
    file = _profile_dir(model, device).joinpath(f"i{input_len}b{batch_size}.json")
    if not file.exists():
        return None
    with open(file, "r") as f:
        data = json.load(f)
    return data


def _get_prefill_latency(data: dict) -> float:
    latencies = sorted(data["prefill_latencies"])
    return sum(latencies[1:-1]) / (len(latencies) - 2)


def _get_decode_latency(data: dict) -> float:
    latencies = sorted(data["decode_latencies"])
    return sum(latencies[1:-1]) / (len(latencies) - 2)


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
        model: ModelSpec,
        device: Optional["DeviceSpec"] = None,
        tp: int = 1,
        training_set: list[Tuple[int, int]] = None,
    ):
        device = _resolve_default_device() if device is None else device
        model_config = get_model_config(model)
        max_model_len = model_config.max_model_len

        self._all_set = [
            (ilen, bs)
            for ilen in [32, 64, 128, 256, 512, 1024, 2048, 4096]
            for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10)))
            if ilen * bs <= max_model_len and (ilen + 10) * bs < 10240
        ]
        self._testing_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 == 0
        ]
        self._training_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 != 0
        ]
        self.model = model
        self.device = device
        self.tp = tp

        X = []
        y = []
        training_set = training_set if training_set is not None else self._training_set
        for input_len, batch_size in training_set:
            t = input_len * batch_size
            s = input_len**2 * batch_size
            if (
                data := _load_time_profile(model, input_len, batch_size, device, tp=tp)
            ) is None:
                continue
            X.append([t, s])
            y.append(_get_prefill_latency(data))

        if not X:
            _raise_missing_profiles(model, device)
        self.linreg = LinearRegression().fit(X, y)

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
        return self.linreg.predict([[t, s]])[0]

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
            if (
                data := _load_time_profile(
                    self.model, input_len, batch_size, self.device, tp=self.tp
                )
            ) is None:
                continue
            X.append([t, s])
            y.append(_get_prefill_latency(data))

        return self.linreg.score(X, y)


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
        model: ModelSpec,
        device: Optional["DeviceSpec"] = None,
        tp: int = 1,
        training_set: list[Tuple[int, int]] = None,
    ):
        device = _resolve_default_device() if device is None else device
        model_config = get_model_config(model)
        max_model_len = model_config.max_model_len

        self._all_set = [
            (ilen, bs)
            for ilen in [32, 64, 128, 256, 512, 1024, 2048, 4096]
            for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10)))
            if ilen * bs <= max_model_len and (ilen + 10) * bs < 10240
        ]
        self._testing_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 == 1
        ]
        self._training_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 != 1
        ]
        self.model = model
        self.device = device
        self.tp = tp

        X = []
        y = []
        training_set = training_set if training_set is not None else self._training_set
        for input_len, batch_size in training_set:
            t = input_len * batch_size
            b = batch_size
            if (
                data := _load_time_profile(model, input_len, batch_size, device, tp=tp)
            ) is None:
                continue
            X.append([t, b])
            y.append(_get_decode_latency(data))

        if not X:
            _raise_missing_profiles(model, device)
        self.linreg = LinearRegression().fit(X, y)

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
            b = batch_size

        return self._predict(t, b)

    def _predict(self, t, b) -> float:
        return self.linreg.predict([[t, b]])[0] * 1.0

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
            if (
                data := _load_time_profile(
                    self.model, input_len, batch_size, self.device, tp=self.tp
                )
            ) is None:
                continue
            X.append([t, b])
            y.append(_get_decode_latency(data))

        return self.linreg.score(X, y)


def test_prefill_estimator(
    model: Optional[ModelSpec] = None, device: Optional[DeviceSpec] = None
):
    if model is None:
        model = ModelType.qwen_7b_chat
    if device is None:
        device = _resolve_default_device()
    est = PrefillEstimator(model, device)
    print(f"Score for {model}, {device} (prefill): {est.test()}")


def test_decode_estimator(
    model: Optional[ModelSpec] = None, device: Optional[DeviceSpec] = None
):
    if model is None:
        model = ModelType.qwen_7b_chat
    if device is None:
        device = _resolve_default_device()
    est = DecodeEstimator(model, device)
    print(f"Score for {model}, {device} (decode): {est.test()}")


_estimators = {}


def make_estimator(
    cls,
    model: Optional[ModelSpec] = None,
    device: Optional[DeviceSpec] = None,
):
    if model is None:
        model = ModelType.qwen_7b_chat
    if device is None:
        device = _resolve_default_device()
    global _estimators
    if (est := _estimators.get((model, device, cls))) is not None:
        return est
    est = cls(model, device)
    _estimators[(model, device, cls)] = est
    return est


def cache_estimators(cls, models: List[ModelSpec], device: DeviceSpec):
    for model in models:
        make_estimator(cls, model, device)


if __name__ == "__main__":
    from aegaeon.model_registry import get_registry

    names = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-4B",
        "llama2_13b_chat",
    ]
    for name in names:
        model = get_registry().get_by_name(name)
        if model is not None:
            test_prefill_estimator(model=model, device=DeviceType.A10)
            test_decode_estimator(model=model, device=DeviceType.A10)
