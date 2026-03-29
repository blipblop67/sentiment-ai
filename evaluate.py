"""
evaluate.py
Evaluate the fine-tuned model on the IMDb test set.
Generates confusion matrix, classification report, and confidence plots.

Usage:
  python evaluate.py --model models/sentiment_bert.pt
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IMDbDataset
from model import load_model, get_tokenizer

CLASSES = ["Negative", "Positive"]


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer   = get_tokenizer()
    test_ds     = IMDbDataset("test", tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    # load_model now correctly moves model to device
    model = load_model(args.model, device=str(device))
    print(f"Loaded model from {args.model}")
    print(f"Evaluating on {len(test_ds):,} test samples...\n")

    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]

            logits = model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=-1).cpu()
            preds  = probs.argmax(dim=-1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Classification report
    print("Classification Report:")
    print("─" * 50)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))

    accuracy = (all_preds == all_labels).mean()
    print(f"Overall accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sentiment Classifier — Model Evaluation", fontsize=14, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=axes[0], linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (counts)")

    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=axes[1], linewidths=0.5,
                cbar_kws={"shrink": 0.8, "format": ticker.FuncFormatter(lambda x, _: f"{x:.0f}%")})
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (% per actual class)")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("\nSaved → confusion_matrix.png")
    plt.show()

    # Confidence distribution
    fig2, ax = plt.subplots(figsize=(8, 4))
    pos_conf = all_probs[all_labels == 1, 1]
    neg_conf = all_probs[all_labels == 0, 0]

    ax.hist(pos_conf, bins=40, alpha=0.6, color="#3498db", label="Positive reviews (confidence)")
    ax.hist(neg_conf, bins=40, alpha=0.6, color="#e74c3c", label="Negative reviews (confidence)")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision threshold 0.5")
    ax.set_xlabel("Model confidence"); ax.set_ylabel("Count")
    ax.set_title("Confidence distribution by actual class")
    ax.legend()
    plt.tight_layout()
    plt.savefig("confidence_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved → confidence_distribution.png")
    plt.show()

    # Error analysis
    wrong_mask  = all_preds != all_labels
    wrong_probs = all_probs[wrong_mask].max(axis=1)
    print(f"\nError analysis:")
    print(f"  Total errors:                  {wrong_mask.sum():,}")
    print(f"  Mean confidence (wrong):       {wrong_probs.mean():.3f}")
    print(f"  High-confidence errors (>0.9): {(wrong_probs > 0.9).sum():,}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate sentiment model")
    p.add_argument("--model",      default="models/sentiment_bert.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
