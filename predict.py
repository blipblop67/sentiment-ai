"""
predict.py
Run sentiment prediction on any text from the terminal.

Usage:
  python predict.py --text "This movie was incredible!"
  python predict.py --interactive
  python predict.py --file reviews.txt
"""

import argparse
import sys

import torch

from model import load_model, get_tokenizer

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}
COLORS = {0: "\033[91m", 1: "\033[92m"}
RESET  = "\033[0m"


def predict_text(text: str, model, tokenizer, device, max_length: int = 256) -> dict:
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=-1).cpu().squeeze()

    pred_class = probs.argmax().item()
    confidence = probs[pred_class].item()

    return {
        "label":      LABELS[pred_class],
        "confidence": confidence,
        "probs":      {"negative": probs[0].item(), "positive": probs[1].item()},
        "class_id":   pred_class,
    }


def format_result(text: str, result: dict) -> str:
    color   = COLORS[result["class_id"]]
    bar_len = int(result["confidence"] * 30)
    bar     = "█" * bar_len + "░" * (30 - bar_len)
    return (
        f"\n  Text:       {text[:80]}{'...' if len(text) > 80 else ''}\n"
        f"  Sentiment:  {color}{result['label']}{RESET}\n"
        f"  Confidence: {bar} {result['confidence']*100:.1f}%\n"
        f"  Negative: {result['probs']['negative']*100:.1f}%  "
        f"Positive: {result['probs']['positive']*100:.1f}%\n"
    )


def run(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(args.model, device=str(device))
    tokenizer = get_tokenizer()

    if args.text:
        result = predict_text(args.text, model, tokenizer, device)
        print(format_result(args.text, result))

    elif args.file:
        with open(args.file) as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"Processing {len(lines)} reviews...\n")
        for line in lines:
            result = predict_text(line, model, tokenizer, device)
            print(format_result(line, result))

    elif args.interactive:
        print("\n── Sentiment Analyser ────────────────────────")
        print("  Type a movie review and press Enter.")
        print("  Type 'quit' to exit.\n")
        while True:
            try:
                text = input("Review: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                sys.exit(0)
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            result = predict_text(text, model, tokenizer, device)
            print(format_result(text, result))
    else:
        print("Provide --text, --file, or --interactive. Use --help for usage.")


def parse_args():
    p = argparse.ArgumentParser(description="Predict sentiment for text")
    p.add_argument("--model",       default="models/sentiment_bert.pt")
    p.add_argument("--text",        default=None)
    p.add_argument("--file",        default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--max-length",  type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
