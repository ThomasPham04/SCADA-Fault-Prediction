"""
Problem definition configuration loader.

Loads the locked CRISP-DM problem configuration from configs/problem_v1.yaml
and exposes typed dataclasses for downstream data preparation and training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBLEM_CONFIG_PATH = PROJECT_ROOT / "configs" / "problem_v1.yaml"


@dataclass(frozen=True)
class TimeSeriesProblemConfig:
    sampling_minutes: int
    input_window_steps: int
    input_window_hours: float
    prediction_horizon_steps: int
    prediction_horizon_hours: float
    stride_steps: int
    stride_minutes: int


@dataclass(frozen=True)
class WindowGenerationConfig:
    pooling_strategy: str
    primary_group_keys: tuple[str, ...]
    processed_group_keys: tuple[str, ...]
    event_id_column_candidates: tuple[str, ...]
    do_not_cross_boundaries: dict[str, bool]
    input_interval: str
    future_label_interval: str


@dataclass(frozen=True)
class LabelConfig:
    row_label_column: str
    positive_label: int
    negative_label: int
    target_name: str
    target_definition: str
    event_label_source: str
    leakage_columns_to_exclude: tuple[str, ...]


@dataclass(frozen=True)
class DatasetScopeConfig:
    root_dir: str
    available_wind_farms: tuple[str, ...]
    initial_scope_wind_farms: tuple[str, ...]
    out_of_scope_wind_farms_v1: tuple[str, ...]
    farm_structure: dict[str, str]
    event_file_unit: str


@dataclass(frozen=True)
class MetricsConfig:
    window_level: tuple[str, ...]
    event_level: tuple[str, ...]
    primary_reporting_level: str


@dataclass(frozen=True)
class ProblemConfig:
    raw: dict[str, Any]
    problem_id: str
    version: str
    task_type: str
    objective_formula: str
    business: dict[str, Any]
    dataset: DatasetScopeConfig
    time_series: TimeSeriesProblemConfig
    window_generation: WindowGenerationConfig
    labels: LabelConfig
    splitting: dict[str, Any]
    metrics: MetricsConfig
    risks: dict[str, Any]
    success_criteria: dict[str, Any]

    @property
    def window_size(self) -> int:
        """Alias for compatibility with existing sequence code."""
        return self.time_series.input_window_steps

    @property
    def horizon(self) -> int:
        """Prediction horizon in timesteps."""
        return self.time_series.prediction_horizon_steps

    @property
    def stride(self) -> int:
        """Window stride in timesteps."""
        return self.time_series.stride_steps


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping section: {key}")
    return value


def _tuple(section: dict[str, Any], key: str) -> tuple[str, ...]:
    value = section.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Expected list for key '{key}', got {type(value).__name__}")
    return tuple(str(item) for item in value)


def _validate_time_series(cfg: TimeSeriesProblemConfig) -> None:
    if cfg.sampling_minutes <= 0:
        raise ValueError("sampling_minutes must be positive.")
    if cfg.input_window_steps <= 0:
        raise ValueError("input_window_steps must be positive.")
    if cfg.prediction_horizon_steps <= 0:
        raise ValueError("prediction_horizon_steps must be positive.")
    if cfg.stride_steps <= 0:
        raise ValueError("stride_steps must be positive.")

    window_hours = cfg.input_window_steps * cfg.sampling_minutes / 60.0
    horizon_hours = cfg.prediction_horizon_steps * cfg.sampling_minutes / 60.0
    stride_minutes = cfg.stride_steps * cfg.sampling_minutes

    if abs(window_hours - float(cfg.input_window_hours)) > 1e-9:
        raise ValueError(
            "input_window_hours does not match input_window_steps * sampling_minutes / 60."
        )
    if abs(horizon_hours - float(cfg.prediction_horizon_hours)) > 1e-9:
        raise ValueError(
            "prediction_horizon_hours does not match prediction_horizon_steps * "
            "sampling_minutes / 60."
        )
    if stride_minutes != cfg.stride_minutes:
        raise ValueError("stride_minutes does not match stride_steps * sampling_minutes.")


def _build_problem_config(raw: dict[str, Any]) -> ProblemConfig:
    problem = _require_mapping(raw, "problem")
    dataset = _require_mapping(raw, "dataset")
    time_series = _require_mapping(raw, "time_series")
    window_generation = _require_mapping(raw, "window_generation")
    labels = _require_mapping(raw, "labels")
    metrics = _require_mapping(raw, "metrics")

    time_series_cfg = TimeSeriesProblemConfig(
        sampling_minutes=int(time_series["sampling_minutes"]),
        input_window_steps=int(time_series["input_window_steps"]),
        input_window_hours=float(time_series["input_window_hours"]),
        prediction_horizon_steps=int(time_series["prediction_horizon_steps"]),
        prediction_horizon_hours=float(time_series["prediction_horizon_hours"]),
        stride_steps=int(time_series["stride_steps"]),
        stride_minutes=int(time_series["stride_minutes"]),
    )
    _validate_time_series(time_series_cfg)

    return ProblemConfig(
        raw=raw,
        problem_id=str(problem["id"]),
        version=str(problem["version"]),
        task_type=str(problem["task_type"]),
        objective_formula=str(problem["objective_formula"]),
        business=_require_mapping(raw, "business"),
        dataset=DatasetScopeConfig(
            root_dir=str(dataset["root_dir"]),
            available_wind_farms=_tuple(dataset, "available_wind_farms"),
            initial_scope_wind_farms=_tuple(dataset, "initial_scope_wind_farms"),
            out_of_scope_wind_farms_v1=_tuple(dataset, "out_of_scope_wind_farms_v1"),
            farm_structure=dict(dataset.get("farm_structure", {})),
            event_file_unit=str(dataset["event_file_unit"]),
        ),
        time_series=time_series_cfg,
        window_generation=WindowGenerationConfig(
            pooling_strategy=str(window_generation["pooling_strategy"]),
            primary_group_keys=_tuple(window_generation, "primary_group_keys"),
            processed_group_keys=_tuple(window_generation, "processed_group_keys"),
            event_id_column_candidates=_tuple(window_generation, "event_id_column_candidates"),
            do_not_cross_boundaries=dict(window_generation.get("do_not_cross_boundaries", {})),
            input_interval=str(window_generation["input_interval"]),
            future_label_interval=str(window_generation["future_label_interval"]),
        ),
        labels=LabelConfig(
            row_label_column=str(labels["row_label_column"]),
            positive_label=int(labels["positive_label"]),
            negative_label=int(labels["negative_label"]),
            target_name=str(labels["target_name"]),
            target_definition=str(labels["target_definition"]),
            event_label_source=str(labels["event_label_source"]),
            leakage_columns_to_exclude=_tuple(labels, "leakage_columns_to_exclude"),
        ),
        splitting=_require_mapping(raw, "splitting"),
        metrics=MetricsConfig(
            window_level=_tuple(metrics, "window_level"),
            event_level=_tuple(metrics, "event_level"),
            primary_reporting_level=str(metrics["primary_reporting_level"]),
        ),
        risks=_require_mapping(raw, "risks"),
        success_criteria=_require_mapping(raw, "success_criteria"),
    )


def load_problem_config(path: str | Path | None = None) -> ProblemConfig:
    """
    Load the locked problem configuration.

    Args:
        path: Optional path to a YAML config. Defaults to configs/problem_v1.yaml.

    Returns:
        ProblemConfig with typed sub-sections and the original raw dictionary.
    """
    config_path = Path(path) if path is not None else DEFAULT_PROBLEM_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Problem config not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load problem YAML configs. "
            "Install dependencies from requirements.txt."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Problem config must contain a YAML mapping: {config_path}")

    return _build_problem_config(raw)


def load_problem_config_dict(path: str | Path | None = None) -> dict[str, Any]:
    """Load the problem config and return its raw dictionary."""
    return load_problem_config(path).raw


if __name__ == "__main__":
    config = load_problem_config()
    print(f"Loaded {config.problem_id} ({config.version})")
    print(
        f"W={config.window_size}, H={config.horizon}, stride={config.stride}, "
        f"farm={', '.join(config.dataset.initial_scope_wind_farms)}"
    )
