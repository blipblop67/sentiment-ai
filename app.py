"""
app.py
Gradio web interface for the sentiment analyser.

Features:
  - BERT sentiment classification (positive / negative)
  - Local LLM movie identification via Ollama (free, no API key)

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model: ollama pull llama3.2
  3. In a separate terminal: ollama serve
  4. Run: python app.py

Share publicly (72-hour link):
  python app.py --share
"""

import argparse
import requests

import gradio as gr
import torch

from model import load_model, get_tokenizer

# Load BERT once at startup
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer()
model     = load_model("models/sentiment_bert.pt", device=str(device))
print(f"BERT model loaded on {device}")

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


# Sentiment prediction (BERT)

def predict_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {"Positive": 0.5, "Negative": 0.5}

    encoding = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=-1).cpu().squeeze()

    return {
        "Positive": float(probs[1]),
        "Negative": float(probs[0]),
    }


# Movie identification (Ollama)

def identify_movie(text: str) -> str:
    if not text or not text.strip():
        return ""

    prompt = (
        f"This is a movie review. Try to identify which movie it is about. "
        f"If you can identify it confidently, state the movie title and year, "
        f"then give a one sentence reason why you think so. "
        f"If you cannot identify the specific movie, say so briefly and mention "
        f"what genre or type of film it seems to be about. "
        f"Keep your response to 2-3 sentences maximum.\n\n"
        f"Review: {text}"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "No response from model.")

    except requests.exceptions.ConnectionError:
        return (
            "Ollama is not running. "
            "Start it with `ollama serve` in a separate terminal, "
            "then refresh the page."
        )
    except Exception as e:
        return f"Error: {str(e)}"


# Combined function

def analyse(text: str):
    return predict_sentiment(text), identify_movie(text)


# Example reviews
EXAMPLES = [
    ["An absolute masterpiece. The lion king's journey and the powerful score made this one of the most emotional experiences I've ever had in a cinema."],
    ["A galaxy far far away has never felt so real. Iconic lightsaber duels and an unforgettable villain in a black mask."],
    ["Complete waste of time. A group of teens in a haunted house making every possible wrong decision. Jump scares every five minutes with zero substance."],
    ["The caped crusader faces his darkest hour in this gritty crime epic. Heath Ledger's performance as the clown prince of crime is utterly terrifying."],
    ["A hobbit's unexpected journey through breathtaking landscapes with dwarves, trolls and a certain precious ring. Peter Jackson at his finest."],
]


# Gradio interface
with gr.Blocks(title="Sentiment Analyser + Movie Guesser", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # Sentiment Analyser + Movie Guesser
        Paste any movie review — the app analyses sentiment using fine-tuned BERT,
        then tries to guess the movie using a local LLM (Ollama).
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Movie review",
                placeholder="Paste or type a movie review here...",
                lines=5,
            )
            submit_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=1):
            sentiment_output = gr.Label(label="Sentiment", num_top_classes=2)

    movie_output = gr.Markdown(label="Movie identification")

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

    gr.Markdown(
        """
        ---
        **Sentiment model:** `bert-base-uncased` fine-tuned on 25,000 IMDb reviews — 92.1% accuracy  
        **Movie identification:** Ollama local LLM (free, runs on your PC)  
        **GitHub:** [github.com/rupa-blip/sentiment-ai](https://github.com/rupa-blip/sentiment-ai)
        """
    )

    submit_btn.click(fn=analyse, inputs=text_input, outputs=[sentiment_output, movie_output])
    text_input.submit(fn=analyse, inputs=text_input, outputs=[sentiment_output, movie_output])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port",  type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.launch(share=args.share, server_port=args.port)
