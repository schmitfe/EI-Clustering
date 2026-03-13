from __future__ import annotations

import argparse
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T", int, float)


def _dedupe_preserve_order(values: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(values))


def _coerce_int(raw: str, *, option_name: str) -> int:
    text = str(raw).strip()
    try:
        return int(text)
    except ValueError:
        try:
            value = float(text)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid integer value '{raw}' for {option_name}.") from exc
        if not value.is_integer():
            raise ValueError(f"Invalid integer value '{raw}' for {option_name}.")
        return int(value)


def _coerce_float(raw: str, *, option_name: str) -> float:
    try:
        return float(str(raw).strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid float value '{raw}' for {option_name}.") from exc


def _iter_range_values(
    start: T,
    stop: T,
    step: T,
    *,
    option_name: str,
    cast: Callable[[float], T],
) -> list[T]:
    step_value = float(step)
    if step_value == 0.0:
        raise ValueError(f"{option_name} range step must be non-zero.")
    start_value = float(start)
    stop_value = float(stop)
    direction = 1.0 if step_value > 0.0 else -1.0
    tolerance = max(abs(step_value), abs(start_value), abs(stop_value), 1.0) * 1e-12
    compare = (lambda current: current <= stop_value + tolerance) if direction > 0.0 else (lambda current: current >= stop_value - tolerance)
    if direction > 0.0 and start_value > stop_value + tolerance:
        raise ValueError(f"{option_name} range start must be <= stop for a positive step.")
    if direction < 0.0 and start_value < stop_value - tolerance:
        raise ValueError(f"{option_name} range start must be >= stop for a negative step.")
    values: list[T] = []
    cursor = start_value
    guard = 0
    while compare(cursor):
        values.append(cast(cursor))
        cursor += step_value
        guard += 1
        if guard > 1_000_000:
            raise ValueError(f"{option_name} range generated too many values.")
    return values


def parse_float_values(
    values: Sequence[str] | None,
    *,
    option_name: str,
    default: Sequence[float] | None = None,
) -> list[float] | None:
    if not values:
        return None if default is None else [float(value) for value in default]
    resolved: list[float] = []
    for raw in values:
        token = str(raw).strip()
        if not token:
            continue
        if ":" in token:
            parts = [part.strip() for part in token.split(":")]
            if len(parts) != 3:
                raise ValueError(f"Invalid range '{raw}' for {option_name}. Use start:stop:step.")
            start, stop, step = (_coerce_float(part, option_name=option_name) for part in parts)
            resolved.extend(_iter_range_values(start, stop, step, option_name=option_name, cast=float))
        else:
            resolved.append(_coerce_float(token, option_name=option_name))
    if not resolved:
        return []
    return _dedupe_preserve_order([round(float(value), 12) for value in resolved])


def parse_int_values(
    values: Sequence[str] | None,
    *,
    option_name: str,
    default: Sequence[int] | None = None,
) -> list[int] | None:
    if not values:
        return None if default is None else [int(value) for value in default]
    resolved: list[int] = []
    for raw in values:
        token = str(raw).strip()
        if not token:
            continue
        if ":" in token:
            parts = [part.strip() for part in token.split(":")]
            if len(parts) != 3:
                raise ValueError(f"Invalid range '{raw}' for {option_name}. Use start:stop:step.")
            start, stop, step = (_coerce_int(part, option_name=option_name) for part in parts)
            resolved.extend(_iter_range_values(start, stop, step, option_name=option_name, cast=lambda value: int(round(value))))
        else:
            resolved.append(_coerce_int(token, option_name=option_name))
    if not resolved:
        return []
    return _dedupe_preserve_order([int(value) for value in resolved])


def resolve_float_values(
    values: Sequence[str] | None,
    *,
    option_name: str,
    default: Sequence[float] | None = None,
    start: float | None = None,
    stop: float | None = None,
    step: float | None = None,
    start_name: str | None = None,
    stop_name: str | None = None,
    step_name: str | None = None,
) -> list[float] | None:
    parsed = parse_float_values(values, option_name=option_name, default=None)
    if parsed:
        return parsed
    if start is None and stop is None and step is None:
        return None if default is None else [float(value) for value in default]
    if start is None or stop is None or step is None:
        start_label = start_name or f"{option_name}-start"
        stop_label = stop_name or f"{option_name}-stop"
        step_label = step_name or f"{option_name}-step"
        raise ValueError(f"Provide {start_label}, {stop_label}, and {step_label} together, or use {option_name}.")
    return _iter_range_values(
        float(start),
        float(stop),
        float(step),
        option_name=option_name,
        cast=float,
    )


def resolve_int_values(
    values: Sequence[str] | None,
    *,
    option_name: str,
    default: Sequence[int] | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    start_name: str | None = None,
    stop_name: str | None = None,
    step_name: str | None = None,
) -> list[int] | None:
    parsed = parse_int_values(values, option_name=option_name, default=None)
    if parsed:
        return parsed
    if start is None and stop is None and step is None:
        return None if default is None else [int(value) for value in default]
    if start is None or stop is None or step is None:
        start_label = start_name or f"{option_name}-start"
        stop_label = stop_name or f"{option_name}-stop"
        step_label = step_name or f"{option_name}-step"
        raise ValueError(f"Provide {start_label}, {stop_label}, and {step_label} together, or use {option_name}.")
    return _iter_range_values(
        int(start),
        int(stop),
        int(step),
        option_name=option_name,
        cast=lambda value: int(round(value)),
    )


def add_v_sweep_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_start: float = 0.0,
    default_stop: float = 1.0,
    default_steps: int = 1000,
) -> None:
    parser.add_argument(
        "--v-range",
        type=str,
        default=f"{default_start:g}:{default_stop:g}:{default_steps}",
        help=(
            "Compact ERF sweep specification as start:stop:steps. "
            f"Defaults to {default_start:g}:{default_stop:g}:{default_steps}."
        ),
    )


def resolve_v_sweep(
    args: argparse.Namespace,
    *,
    range_attr: str = "v_range",
) -> tuple[float, float, int]:
    raw_range = getattr(args, range_attr, None)
    if not raw_range:
        raise ValueError("Missing --v-range value.")
    parts = [part.strip() for part in str(raw_range).split(":")]
    if len(parts) != 3:
        raise ValueError("Invalid --v-range value. Use start:stop:steps.")
    start = _coerce_float(parts[0], option_name="--v-range")
    stop = _coerce_float(parts[1], option_name="--v-range")
    steps = _coerce_int(parts[2], option_name="--v-range")
    if steps <= 0:
        raise ValueError("--v-range must use a positive step count.")
    return float(start), float(stop), int(steps)
