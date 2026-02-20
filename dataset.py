from datasets import Features, Value, load_dataset
from torch.utils.data import DataLoader

# Login using e.g. `huggingface-cli login` to access this dataset
features = Features(
    {
        "source": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "rating": Value("int64"),
    }
)

ds = load_dataset("sapientinc/sudoku-extreme", features=features)
print(ds["train"][0])


def collate_fn(batch):
    return {
        "source": [item["source"] for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "rating": [item["rating"] for item in batch],
    }


dataloader = DataLoader(ds["train"], batch_size=32, collate_fn=collate_fn, shuffle=True)

for batch in dataloader:
    print(batch)
    break
    # Print only question and answer from the batch
    for key in ["question", "answer"]:
        print(f"{key}: {batch[key]}")
