from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import yaml  # type: ignore
except ModuleNotFoundError as _yaml_error:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]
else:
    _yaml_error = None


CONFIG_DIR = Path(__file__).resolve().parent


def _to_human(obj, *, float_precision=6, nan_policy="string"):
    if isinstance(obj, np.ndarray):
        return [
            _to_human(x, float_precision=float_precision, nan_policy=nan_policy)
            for x in obj.tolist()
        ]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()
    if obj is np.nan:
        return "NaN" if nan_policy == "string" else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN" if nan_policy == "string" else None
        if math.isinf(obj):
            return ("Infinity" if obj > 0 else "-Infinity") if nan_policy == "string" else None
        return round(obj, float_precision) if float_precision is not None else obj
    if isinstance(obj, (list, tuple)):
        return [
            _to_human(x, float_precision=float_precision, nan_policy=nan_policy)
            for x in obj
        ]
    if isinstance(obj, dict):
        return {
            k: _to_human(v, float_precision=float_precision, nan_policy=nan_policy)
            for k, v in obj.items()
        }
    return str(obj)


def write_human_json(
    cfg: dict,
    path: str | Path,
    *,
    float_precision: int = 6,
    nan_policy: str = "string",
    ensure_ascii: bool = False,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    human = _to_human(cfg, float_precision=float_precision, nan_policy=nan_policy)
    txt = json.dumps(human, indent=2, sort_keys=True, ensure_ascii=ensure_ascii)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(path)

def write_yaml_config(cfg: dict, path: str | Path) -> None:
    if yaml is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to write configuration files. Install it via 'pip install pyyaml'."
        ) from _yaml_error
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)


def _normalize_for_tag(obj):
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return {"__float__": "NaN"}
        if math.isinf(obj):
            return {"__float__": "Infinity" if obj > 0 else "-Infinity"}
        return {"__float__": format(obj, ".17g")}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_tag(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_for_tag(obj[k]) for k in sorted(obj.keys())}
    return str(obj)


def sim_tag_from_cfg(cfg: dict, *, length: int = 10) -> str:
    canon = _normalize_for_tag(cfg)
    blob = json.dumps(canon, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def load_config(
    name: str = "default_simulation",
    *,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    config_path = _resolve_config_path(name)
    if yaml is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to load configuration files. Install it via 'pip install pyyaml'."
        ) from _yaml_error
    with config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    if not overrides:
        return base_config
    override_dict = parse_overrides(overrides)
    return deep_update(base_config, override_dict)


def resolve_base_config(descriptor: Any) -> dict[str, Any]:
    if descriptor is None:
        raise ValueError("Base configuration descriptor must not be None.")
    if isinstance(descriptor, Mapping):
        return deepcopy(descriptor)
    if isinstance(descriptor, (str, Path)):
        return load_config(str(descriptor))
    raise TypeError(
        "Base configuration must be provided as a mapping or a config name/path "
        f"(got {type(descriptor).__name__})."
    )


def first_float(
    value: Any,
    *,
    cell_type: str | None = None,
    default: float | None = None,
) -> float:
    if value is None:
        if default is None:
            raise ValueError("No value available to coerce into float.")
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        if cell_type and cell_type in value:
            return first_float(value[cell_type], cell_type=cell_type, default=default)
        if "default" in value:
            return first_float(value["default"], cell_type=cell_type, default=default)
        if "excitatory" in value:
            return first_float(value["excitatory"], cell_type=cell_type, default=default)
        first_key = next(iter(value))
        return first_float(value[first_key], cell_type=cell_type, default=default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            if default is None:
                raise ValueError("Cannot extract float from empty sequence.")
            return float(default)
        return float(value[0])
    raise TypeError(
        "Expected a float-like value, sequence, or mapping; "
        f"got {type(value).__name__}."
    )


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Override '{raw}' is missing '='.")
        key_path, raw_value = raw.split("=", 1)
        target = root
        *parents, leaf = key_path.split(".")
        for segment in parents:
            target = target.setdefault(segment, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot override non-dict path '{key_path}'.")
        target[leaf] = yaml.safe_load(raw_value)
    return root


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.suffix:
        return candidate if candidate.is_file() else CONFIG_DIR / candidate
    return CONFIG_DIR / f"{name}.yaml"


def add_override_arguments(
    parser: argparse.ArgumentParser,
    *,
    config_option: str = "--config",
    overwrite_option: str = "--overwrite",
    config_default: str = "default_simulation",
    overwrite_metavar: str = "path=value",
) -> None:
    parser.add_argument(
        config_option,
        default=config_default,
        help="Config name or path (defaults to '%(default)s').",
    )
    parser.add_argument(
        "-O",
        overwrite_option,
        action="append",
        default=[],
        metavar=overwrite_metavar,
        help="Override a config value using dotted-path notation (may be repeated).",
    )


def load_from_args(
    args: argparse.Namespace,
    *,
    config_attr: str = "config",
    overwrite_attr: str = "overwrite",
    default_config: str = "default_simulation",
) -> dict[str, Any]:
    config_name = getattr(args, config_attr, default_config) or default_config
    overrides = getattr(args, overwrite_attr, None)
    return load_config(config_name, overrides=overrides)


__all__ = [
    "_to_human",
    "write_human_json",
    "write_yaml_config",
    "_normalize_for_tag",
    "sim_tag_from_cfg",
    "load_config",
    "parse_overrides",
    "deep_update",
    "resolve_base_config",
    "first_float",
    "CONFIG_DIR",
    "add_override_arguments",
    "load_from_args",
]
