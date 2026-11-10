"""Datasets: ImageNet val, Imagenette and WikiText-2.

A loader is a callable that returns a DataLoader. A config names it in `dataset.loader`.
Set `loader.eval_mode` to "classification", "next_token" or "perplexity".
The `evaluation.mode` of a config overrides it.

Batch format (see `accuracim.evaluation.run_batch`):
- CNNs: `(inputs, targets)`
- LLMs: dict with `input_ids`/`labels`

NOTE: Wrap new datasets with `get_deterministic_dataloader`. Simulation compares a noisy run
with a cached ideal run, batch by batch. The two runs must see the same batches.
"""

import torch
import json
import os

# Image datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# LLM datasets
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Tokenizer, DefaultDataCollator

from accuracim.config import project_root
from accuracim.dataloader import get_deterministic_dataloader

# HF tokenizer for the LLM dataloader
MODEL_NAME_HF = "gpt2"

# ==============================================================================
# IMAGENET DATASETS
# ==============================================================================


class CustomImageNetDataset(Dataset):
    """ImageNet in `root_dir/<synset>/<image>` folders, labeled via `imagenet_class_index.json`."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(os.path.join(project_root(), "imagenet_class_index.json")) as f:
            idx_to_label = json.load(f)

        # synset -> class index
        self.synset_to_idx = {v[0]: int(k) for k, v in idx_to_label.items()}

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            label = self.synset_to_idx.get(class_name, None)
            if label is None:
                continue

            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(class_path, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNetValDataset(Dataset):
    """ImageNet val in a flat layout (the extracted `ILSVRC2012_img_val.tar`).

    `root_dir` holds the images; `labels_file` (default:
    `workloads/data/imagenet_val_labels.txt`) has one `<filename> <0-based label>` per line.
    """

    def __init__(self, root_dir, labels_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if labels_file is None:
            labels_file = os.path.join(os.path.dirname(__file__), "data", "imagenet_val_labels.txt")

        with open(labels_file, "r") as f:
            labels = dict(line.split() for line in f if line.strip())

        # Sorted filenames: sample order feeds the seeded permutation, so it keys the cache.
        self.samples = [
            (os.path.join(root_dir, fname), int(labels[fname])) for fname in sorted(labels)
        ]

        missing = [p for p, _ in self.samples if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.samples)} images missing from {root_dir}, e.g. {missing[0]}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================================================================
# DATALOADER GETTERS
# ==============================================================================


def get_imagenet_dataloader(
    batch_size=4, num_workers=4, deterministic_dataloader=False, size="Large", root=None
):
    """Return an ImageNet val DataLoader.

    `size`: "Small" (5k, `<root>/val`) or "Large" (50k, `root`, else `$IMAGENET_VAL_DIR`).
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if "s" in size or "S" in size:
        # NOTE: adjust to your ImageNet val directory
        val_dir = os.path.join(project_root(), "val")
        dataset = CustomImageNetDataset(val_dir, transform=transform)
    elif "l" in size or "L" in size:
        root_dir = root or os.environ.get("IMAGENET_VAL_DIR")
        if not root_dir:
            raise FileNotFoundError(
                "Set IMAGENET_VAL_DIR (or dataset.params.root) to the extracted ILSVRC2012_img_val.tar folder."
            )
        dataset = ImageNetValDataset(os.path.expanduser(root_dir), transform=transform)
    else:
        raise ValueError("Invalid size specified. Use 'Small (5k)' or 'Large (50k)'.")

    if deterministic_dataloader:
        dataloader = get_deterministic_dataloader(dataset, batch_size, 42, num_workers)
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    dataloader.eval_mode = "classification"  # metric for this dataset (see accuracim.evaluation)
    return dataloader


#: Imagenette's 10 classes -> their ImageNet-1k indices (and torchvision category names,
#: checked at load time so a wrong index fails loudly)
IMAGENETTE_TO_IMAGENET = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}


class _ToImageNetIndex:
    """Imagenette class index -> ImageNet-1k index (a class, so worker processes can pickle it)."""

    def __init__(self, wnids):
        unknown = [w for w in wnids if w not in IMAGENETTE_TO_IMAGENET]
        if unknown:
            raise ValueError(
                f"Imagenette classes {unknown} are missing from IMAGENETTE_TO_IMAGENET"
            )
        self.indices = [IMAGENETTE_TO_IMAGENET[w][0] for w in wnids]

    def __call__(self, target):
        return self.indices[target]


def get_imagenette_dataloader(
    batch_size=32, num_workers=4, deterministic_dataloader=True, root=None, size="320px"
):
    """Return an Imagenette val DataLoader: 3,925 images of 10 ImageNet classes, labeled
    with their ImageNet-1k indices so ImageNet models run unchanged.

    Downloads to `root` (default `<project root>/data`) on first use (torchvision >= 0.17).
    Imagenette: https://github.com/fastai/imagenette
    """
    from torchvision.datasets import Imagenette
    from torchvision.models import ResNet18_Weights

    categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    for wnid, (index, name) in IMAGENETTE_TO_IMAGENET.items():
        if categories[index] != name:
            raise ValueError(
                f"{wnid}: ImageNet index {index} is {categories[index]!r}, not {name!r}"
            )

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    root = os.path.expanduser(root or os.path.join(project_root(), "data"))
    try:
        dataset = Imagenette(root, split="val", size=size, transform=transform)
    except RuntimeError:  # not downloaded yet (torchvision refuses to re-download)
        dataset = Imagenette(root, split="val", size=size, transform=transform, download=True)
    wnids = getattr(dataset, "wnids", None) or sorted(IMAGENETTE_TO_IMAGENET)
    dataset.target_transform = _ToImageNetIndex(wnids)

    if deterministic_dataloader:
        dataloader = get_deterministic_dataloader(dataset, batch_size, 42, num_workers)
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    dataloader.eval_mode = "classification"  # metric for this dataset (see accuracim.evaluation)
    return dataloader


def get_llm_dataloader(
    batch_size: int = 8,
    num_workers: int = 4,
    deterministic_dataloader: bool = False,
    block_size: int = 128,
    tokenizer: str = MODEL_NAME_HF,
):
    """Return a WikiText-2 val DataLoader, tokenized into `block_size` chunks.

    `tokenizer`: Hugging Face id of the model's tokenizer (default GPT-2). Use the same id
    as the model's `hf_id` in `workloads/models.py`.
    """
    if tokenizer == MODEL_NAME_HF:
        tokenizer_ = GPT2Tokenizer.from_pretrained(MODEL_NAME_HF)
        special_tokens = {}  # GPT-2 adds none; unchanged from before
    else:
        tokenizer_ = AutoTokenizer.from_pretrained(tokenizer)
        # Text is concatenated into blocks: no BOS/EOS inserted at every line
        special_tokens = {"add_special_tokens": False}
    tokenizer = tokenizer_
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_examples(examples):
        return tokenizer(examples["text"], return_attention_mask=True, **special_tokens)

    tokenized = raw.map(tokenize_examples, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # Concatenate, then split into `block_size` blocks (remainder dropped).
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)
    eval_dataset = lm_datasets["validation"]

    data_collator = DefaultDataCollator(return_tensors="pt")

    if deterministic_dataloader:
        dataloader = get_deterministic_dataloader(
            dataset=eval_dataset,
            batch_size=batch_size,
            seed=42,
            num_workers=num_workers,
            collate_fn=data_collator,
        )
    else:
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=num_workers,
        )

    dataloader.eval_mode = "next_token"  # metric for this dataset (see accuracim.evaluation)
    return dataloader


# ==============================================================================
# EXAMPLE USAGE: determinism check
# ==============================================================================

if __name__ == "__main__":
    print("--- Testing ImageNet Dataloaders ---")

    print("Getting deterministic ImageNet dataloader...")
    deterministic_img_loader = get_imagenet_dataloader(
        batch_size=4, num_workers=2, deterministic_dataloader=True
    )
    batch_img_1, batch_labels_1 = next(iter(deterministic_img_loader))

    # Rebuild; first batch must match.
    deterministic_img_loader_2 = get_imagenet_dataloader(
        batch_size=4, num_workers=2, deterministic_dataloader=True
    )
    batch_img_2, batch_labels_2 = next(iter(deterministic_img_loader_2))

    if torch.equal(batch_labels_1, batch_labels_2):
        print(
            f"Success: ImageNet dataloader is deterministic. First batch labels: {batch_labels_1.tolist()}"
        )
    else:
        print("Failure: ImageNet dataloader is NOT deterministic.")

    print("\n" + "=" * 50 + "\n")

    print("--- Testing LLM Dataloaders ---")

    print("Getting deterministic LLM dataloader...")
    deterministic_llm_loader = get_llm_dataloader(
        batch_size=2, num_workers=2, deterministic_dataloader=True
    )
    batch_llm_1 = next(iter(deterministic_llm_loader))

    # Rebuild; first batch must match.
    deterministic_llm_loader_2 = get_llm_dataloader(
        batch_size=2, num_workers=2, deterministic_dataloader=True
    )
    batch_llm_2 = next(iter(deterministic_llm_loader_2))

    if torch.equal(batch_llm_1["input_ids"], batch_llm_2["input_ids"]):
        print(
            f"Success: LLM dataloader is deterministic. Shape of first batch: {batch_llm_1['input_ids'].shape}"
        )
    else:
        print("Failure: LLM dataloader is NOT deterministic.")
