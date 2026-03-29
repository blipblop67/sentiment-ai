"""
dataset.py
Load and preprocess the IMDb movie review dataset.

The IMDb dataset contains 50,000 movie reviews labelled as:
  1 = positive
  0 = negative

Uses Hugging Face `datasets` to download automatically —
no manual file handling needed.
"""

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class IMDbDataset(Dataset):
    """
    PyTorch Dataset wrapping the Hugging Face IMDb dataset.

    Each item returns a dict with:
      input_ids      : token IDs       (max_length,)
      attention_mask : padding mask    (max_length,)
      label          : 0 or 1          scalar
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        print(f"Loading IMDb {split} split...")
        raw = load_dataset("imdb", split=split)

        self.texts      = raw["text"]
        self.labels     = raw["label"]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          self.labels[idx],
        }


def load_imdb(tokenizer: PreTrainedTokenizer, max_length: int = 256):
    """Returns (train_dataset, test_dataset)."""
    return (
        IMDbDataset("train", tokenizer, max_length),
        IMDbDataset("test",  tokenizer, max_length),
    )


if __name__ == "__main__":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds, test_ds = load_imdb(tokenizer)
    print(f"Train: {len(train_ds):,} samples")
    print(f"Test:  {len(test_ds):,} samples")
    sample = train_ds[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['label']} ({'positive' if sample['label'] == 1 else 'negative'})")
