"""
model.py
Fine-tunable BERT-based sentiment classifier.

Architecture:
  Pretrained BERT (bert-base-uncased, 110M params)
      │
      ▼
  [CLS] token representation  (768-dim)
      │
      ▼
  Dropout (0.3)
      │
      ▼
  Linear(768 → 2)
      │
      ▼
  Softmax → [P(negative), P(positive)]

The [CLS] token is used because BERT is trained to encode
sentence-level meaning into it — ideal for classification.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class SentimentClassifier(nn.Module):
    """BERT fine-tuned for binary sentiment classification."""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            logits: (batch, 2)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.classifier(cls_output)               # (batch, 2)

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns softmax probabilities instead of raw logits."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)


def load_model(path: str, device: str = "cpu") -> SentimentClassifier:
    """Load a saved model checkpoint and move it to the correct device."""
    model = SentimentClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)   # ← ensures model and data are on same device
    model.eval()
    return model


def get_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizer:
    """Load the matching BERT tokenizer."""
    return BertTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    model = SentimentClassifier()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    tokenizer = get_tokenizer()
    inputs = tokenizer(
        ["This movie was absolutely fantastic!",
         "Worst film I have ever seen."],
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    logits = model(inputs["input_ids"], inputs["attention_mask"])
    probs  = torch.softmax(logits, dim=-1)
    print(f"\nSample output (untrained):")
    for i, text in enumerate(["Positive review", "Negative review"]):
        print(f"  {text}: neg={probs[i,0]:.3f}  pos={probs[i,1]:.3f}")
