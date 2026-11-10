"""Deterministic dataloading. Wrap every dataset with `get_deterministic_dataloader`.

Simulation compares a noisy run with a cached ideal run, batch by batch.
The two runs must see the same batches.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def get_deterministic_dataloader(dataset, batch_size, seed, num_workers, collate_fn=None):
    """Return a DataLoader over a seed-fixed permutation (PyTorch or HF datasets)."""
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g).tolist()

    subset = Subset(dataset, indices)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,  # order fixed by Subset
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )
    return loader
