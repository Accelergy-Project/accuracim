"""Logging: console output, a log file per run, and shared number formats.

Each run logs to the console (INFO) and to `logs/<model>/<timestamp>_<name>.log` (DEBUG).
In the package, use `logging.getLogger(__name__)`. Outside the package, use `get_logger`.

Number formats:
- metrics (accuracy, flips, cross-entropy, perplexity): 4 decimals, not in %
- noise (mean squared output error): 4 significant digits
- accuracy loss: %, 2 decimals
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from contextlib import contextmanager

LOGGER_NAME = "accuracim"
_WIDTH = 78
_indent = 0  # nesting depth, see `indented`


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `accuracim` namespace (for code outside the package)."""
    if name == LOGGER_NAME or name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


log = logging.getLogger(LOGGER_NAME)


@contextmanager
def indented():
    """Indent everything logged inside the block one level (nested work, e.g. anchor runs)."""
    global _indent
    _indent += 1
    try:
        yield
    finally:
        _indent -= 1


class _Formatter(logging.Formatter):
    """Applies the current indent; tags warnings and errors with their level."""

    def __init__(self, fmt: str, datefmt: str, tag_level: bool):
        super().__init__(fmt, datefmt)
        self.tag_level = tag_level

    def format(self, record: logging.LogRecord) -> str:
        record.short_name = record.name.removeprefix(LOGGER_NAME + ".")
        message = record.getMessage().rstrip()
        if self.tag_level and record.levelno >= logging.WARNING:
            message = f"{record.levelname}: {message}"
        record.indented_message = "  " * _indent + message
        return super().format(record)


_CONSOLE_FORMAT = _Formatter("%(asctime)s %(indented_message)s", "%H:%M:%S", tag_level=True)
_FILE_FORMAT = _Formatter(
    "%(asctime)s %(levelname)-7s %(short_name)-12s %(indented_message)s",
    "%Y-%m-%d %H:%M:%S",
    tag_level=False,
)

_file_path: str | None = None


def _attach(handler: logging.Handler) -> None:
    # Our logger and Python warnings (e.g. numpy RuntimeWarnings) share the handlers
    for name in (LOGGER_NAME, "py.warnings"):
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    logging.captureWarnings(True)


def setup_console(verbose: bool = False) -> None:
    """Log to the console (stderr): INFO, or DEBUG with `verbose`. Idempotent."""
    if any(getattr(h, "_accuracim_console", False) for h in log.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(_CONSOLE_FORMAT)
    handler._accuracim_console = True
    _attach(handler)


def setup_file(path: str, stream=None) -> str:
    """Log everything (DEBUG) to `path`, or to an already open `stream` writing it.

    Only the first call takes effect; later calls return the path already in use.
    """
    global _file_path
    if _file_path is not None:
        return _file_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if stream is None:
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    else:
        handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_FILE_FORMAT)
    _attach(handler)
    _file_path = path
    return path


def default_log_path(root: str, model_name: str, run_name: str | None) -> str:
    """Return `<root>/logs/<model>/<timestamp>[_<run_name>].log` (timestamp first, as results)."""
    stem = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        stem += f"_{run_name}"
    return os.path.join(root, "logs", model_name, f"{stem}.log")


# --- Formats -------------------------------------------------------------------


def section(title: str) -> None:
    """Log a phase header: `── title ─────────`."""
    log.info(f"── {title} ".ljust(_WIDTH, "─"))


def fmt_metric(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def fmt_noise(value) -> str:
    return "—" if value is None else f"{value:.4g}"


def fmt_metrics(metrics: dict) -> str:
    """`top1 0.3507 · top5 0.5792 · top1_flip 0.0172` (`count` and the like skipped)."""
    return " · ".join(
        f"{name} {fmt_metric(value)}"
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and name != "count"
    )


def fmt_duration(seconds: float) -> str:
    """`8.2s`, `3m05s`, `2h04m`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def short_path(path: str) -> str:
    """`path` relative to the working directory when inside it, else as given."""
    try:
        rel = os.path.relpath(path)
    except ValueError:  # other drive (Windows)
        return path
    return path if rel.startswith("..") else rel


def source_label(noise_source: str) -> str:
    """`adc_physical_Wbits-10_..._StateIndependentNoise` -> `adc_physical (StateIndependentNoise)`."""
    parts = noise_source.split("_")
    label = "_".join(parts[:2])
    if len(parts) > 2 and parts[-1][:1].isupper():
        label += f" ({parts[-1]})"
    return label


def swept_keys(params_list: list[dict]) -> list[str]:
    """Params keys whose value differs across `params_list` (what a sweep varies)."""
    keys = []
    for params in params_list:
        for key in params:
            if key not in keys:
                keys.append(key)
    return [k for k in keys if len({repr(p.get(k)) for p in params_list}) > 1]


def describe_params(params: dict, keys: list[str]) -> str:
    """`key=value, ...` for `keys` (e.g. the swept ones)."""
    return ", ".join(f"{k}={params.get(k)}" for k in keys)


def log_table(columns: list[str], rows: list[list], level: int = logging.INFO) -> None:
    """Log an aligned table, one line per row."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [max([len(str(col))] + [len(r[i]) for r in cells]) for i, col in enumerate(columns)]
    log.log(level, "  ".join(str(col).ljust(w) for col, w in zip(columns, widths)).rstrip())
    for row in cells:
        log.log(level, "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())


def log_results(
    kind: str,
    params_list: list[dict],
    metrics: dict,
    noises: dict[str, list] | None = None,
    max_console_rows: int = 50,
) -> None:
    """Log one row per params set: swept params, noise, metrics. Full params go to DEBUG.

    Longer than `max_console_rows`: the table goes to the log file only.
    """
    section(f"Results ({kind})")
    keys = swept_keys(params_list)
    noises = noises or {}
    metric_names = [m for m in metrics if m != "count"]
    columns = ["#"] + keys + [("noise" if pl == "__output__" else f"noise[{pl}]") for pl in noises]
    columns += metric_names

    def value_at(values, i):
        if isinstance(values, (list, tuple)):
            return values[i] if i < len(values) else None
        return values

    rows = []
    for i, params in enumerate(params_list):
        row = [i + 1] + [params.get(k) for k in keys]
        row += [fmt_noise(value_at(v, i)) for v in noises.values()]
        row += [fmt_metric(value_at(metrics[m], i)) for m in metric_names]
        rows.append(row)
        log.debug(f"params {i + 1}: {params}")
    if len(rows) > max_console_rows:
        log.info(f"{len(rows)} params sets: full table in the log file and the results workbook")
        log_table(columns, rows, level=logging.DEBUG)
    else:
        log_table(columns, rows)
