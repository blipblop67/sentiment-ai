"""
train.py
Fine-tune BERT on the IMDb sentiment dataset.

Usage:
  python train.py --epochs 3 --batch-size 16   # local
  python train.py --epochs 3 --batch-size 32   # Colab GPU
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import IMDbDataset
from model import SentimentClassifier, get_tokenizer


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cpu":
        print("  Note: BERT on CPU is slow (~1-2 hrs/epoch).")
        print("  Use Google Colab for free GPU.\n")

    # Tokenizer + datasets
    tokenizer  = get_tokenizer()
    full_train = IMDbDataset("train", tokenizer, max_length=args.max_length)

    val_size   = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,            num_workers=0)

    # Model — different LR for BERT vs classifier head
    model = SentimentClassifier(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(),       "lr": args.bert_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc="  Training"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(dim=-1) == labels).sum().item()
            total      += len(labels)

        train_acc  = correct / total
        train_loss = total_loss / total

        # Validate
        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Validating"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                logits        = model(input_ids, attention_mask)
                val_loss_sum += criterion(logits, labels).item() * len(labels)
                val_correct  += (logits.argmax(dim=-1) == labels).sum().item()
                val_total    += len(labels)

        val_acc  = val_correct / val_total
        val_loss = val_loss_sum / val_total

        print(f"  train loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  val   loss={val_loss:.4f}  acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (val acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BERT for sentiment analysis")
    p.add_argument("--save",       default="models/sentiment_bert.pt")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--max-length", type=int,   default=256)
    p.add_argument("--bert-lr",    type=float, default=2e-5)
    p.add_argument("--head-lr",    type=float, default=1e-4)
    p.add_argument("--dropout",    type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
