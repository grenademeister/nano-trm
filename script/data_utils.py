import os

import torch
from datasets import Features, Value, load_dataset, load_from_disk
from torch.utils.data import DataLoader

features = Features(
    {
        "source": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "rating": Value("int64"),
    }
)


def collate_fn(batch):
    return {
        "source": [item["source"] for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "rating": [item["rating"] for item in batch],
    }


def encode_row(row):
    values = []
    for ch in str(row):
        if ch == ".":
            values.append(0)
        elif ch.isdigit():
            values.append(int(ch))
    if len(values) != 81:
        raise ValueError(f"Expected 81 cells, got {len(values)} from row={row!r}")
    return values


def batch_to_tokens(batch, device):
    question_tokens = torch.tensor(
        [encode_row(q) for q in batch["question"]],
        dtype=torch.long,
        device=device,
    )
    answer_tokens = torch.tensor(
        [encode_row(a) for a in batch["answer"]],
        dtype=torch.long,
        device=device,
    )
    return question_tokens, answer_tokens


def get_or_create_cached_splits(data_cfg):
    cache_dir = data_cfg["cache_dir"]
    train_cache_size = data_cfg["train_cache_size"]
    val_cache_size = data_cfg["val_cache_size"]
    dataset_name = data_cfg["dataset_name"]
    test_size = data_cfg["test_size"]
    split_seed = data_cfg["split_seed"]

    if os.path.isdir(os.path.join(cache_dir, "train")) and os.path.isdir(
        os.path.join(cache_dir, "validation")
    ):
        train_dataset = load_from_disk(os.path.join(cache_dir, "train"))
        val_dataset = load_from_disk(os.path.join(cache_dir, "validation"))
        print(f"Loaded cached dataset from {cache_dir}")
        return train_dataset, val_dataset

    full_train = load_dataset(dataset_name, features=features)["train"]
    split = full_train.train_test_split(test_size=test_size, seed=split_seed)
    train_dataset = split["train"].select(range(min(train_cache_size, len(split["train"]))))
    val_dataset = split["test"].select(range(min(val_cache_size, len(split["test"]))))

    os.makedirs(cache_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(cache_dir, "train"))
    val_dataset.save_to_disk(os.path.join(cache_dir, "validation"))
    print(f"Saved cached dataset to {cache_dir}")
    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size, device):
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader
