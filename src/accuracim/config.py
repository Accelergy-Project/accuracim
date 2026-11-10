"""Load configs and resolve their names to callables.

A config (YAML, JSON or Python) names a model, a dataset, a metric and a hardware params grid.
Names resolve through `REGISTRY` (built-in names, e.g. "imagenet", "llm")
or through a "module.sub:func" path.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, Callable


def project_root() -> str:
    """Return the experiment data root: `CIM_ROOT` if set, else cwd (resolved per call).

    Not derived from `__file__`, which would point into site-packages when installed
    and make separate checkouts share one cache.
    """
    return os.path.abspath(os.environ.get("CIM_ROOT") or os.getcwd())


def results_path(model_name: str, category: str, run_name: str | None, N: int | None) -> str:
    """Return `results/<model>/<category>/<timestamp>[_<run_name>][_N<N>].xlsx`, creating the dir.

    Timestamp leads so `ls` sorts chronologically.
    """
    directory = os.path.join(project_root(), "results", model_name, category)
    os.makedirs(directory, exist_ok=True)

    stem = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        stem += f"_{run_name}"
    if N is not None:
        stem += f"_N{N}"
    return os.path.join(directory, f"{stem}.xlsx")


def _load_from_py(path: str) -> dict[str, Any]:
    spec = spec_from_file_location("cim_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load python config: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "CONFIG"):
        return getattr(module, "CONFIG")
    if hasattr(module, "EXPERIMENT"):
        return getattr(module, "EXPERIMENT")
    raise ValueError("Python config must define CONFIG or EXPERIMENT dict.")


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML, JSON, or Python config.

    `name` defaults to the file stem so results trace back to their config.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to load YAML configs.") from exc
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    elif ext == ".py":
        config = _load_from_py(path)
    else:
        raise ValueError(f"Unsupported config extension: {ext}")

    config.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    config.setdefault("config_path", path)  # shown in the log header
    return config


def import_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a "module.sub:func" path."""
    if ":" not in path:
        raise ValueError("Callable path must be in the form 'module.sub:func'.")
    module_path, attr_name = path.split(":", 1)
    module = import_module(module_path)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(f"Callable '{attr_name}' not found in '{module_path}'.") from exc


@dataclass
class Registry:
    name: str
    items: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, key: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.items[key] = fn
        return fn

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self.items:
            available = ", ".join(sorted(self.items.keys()))
            raise KeyError(f"Unknown {self.name} '{key}'. Available: [{available}]")
        return self.items[key]

    def resolve(self, spec: Any) -> Callable[..., Any]:
        """Resolve a callable, registry key, {"name"|"path": ...} dict, or "module:func" path."""
        if spec is None:
            raise ValueError(f"Missing {self.name} spec.")
        if callable(spec):
            return spec
        if isinstance(spec, dict):
            if "name" in spec:
                return self.get(spec["name"])
            if "path" in spec:
                return import_callable(spec["path"])
        if isinstance(spec, str):
            if spec in self.items:
                return self.get(spec)
            if ":" in spec:
                return import_callable(spec)
        raise ValueError(f"Unrecognized {self.name} spec: {spec}")


@dataclass
class RegistryBundle:
    models: Registry = field(default_factory=lambda: Registry("model"))
    datasets: Registry = field(default_factory=lambda: Registry("dataset"))
    evaluations: Registry = field(default_factory=lambda: Registry("evaluation"))


REGISTRY = RegistryBundle()
