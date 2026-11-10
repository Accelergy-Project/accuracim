"""GPU selection and memory cleanup."""

import gc
import logging
import os

import numpy as np
import pynvml
import torch

logger = logging.getLogger(__name__)


def free_memory(garbage_collect=True, cuda_empty_cache=True):
    """Run `gc.collect()` and empty the CUDA cache (if available)."""
    if garbage_collect:
        gc.collect()
    if cuda_empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _visible_gpu_handles() -> list:
    """NVML handles of the GPUs torch can see, in torch's index order.

    NVML lists every GPU in the machine; torch sees only `CUDA_VISIBLE_DEVICES`
    (indices or UUIDs), renumbered from 0.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    handles = []
    for entry in (e.strip() for e in visible.split(",")):
        if not entry or entry.startswith("-"):
            break  # CUDA stops at the first invalid entry
        if entry.isdigit():
            handles.append(pynvml.nvmlDeviceGetHandleByIndex(int(entry)))
        else:
            handles.append(pynvml.nvmlDeviceGetHandleByUUID(entry))
    return handles


def get_best_gpu():
    """Return "cuda:<id>" for the visible GPU with the most free memory, else "cpu"."""
    try:
        pynvml.nvmlInit()
        handles = _visible_gpu_handles()
        if not handles:
            return "cpu"

        free_memory = [pynvml.nvmlDeviceGetMemoryInfo(h).free for h in handles]
        best_gpu_id = int(np.argmax(free_memory))
        logger.debug(f"Found {len(handles)} GPUs; selecting GPU {best_gpu_id} (most free memory)")
        return f"cuda:{best_gpu_id}"

    except pynvml.NVMLError:
        logger.debug("NVIDIA driver not found; using the CPU")
        return "cpu"
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass  # no driver, nothing to shut down


def get_device(model=None, build_model=None):
    """Return the device of `model`, or ask `build_model` for it; None if neither."""
    if model is not None:
        return next(model.parameters()).device
    elif build_model:
        return build_model(return_device=True)
    return None
