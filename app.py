from pathlib import Path
import json

import gradio as gr
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from train import EMBEDDING_MODEL_NAME, MODEL_DIR, train


MODEL_PATH = MODEL_DIR / "sentiment_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DATA_PATH = Path("data/student_feedback_long.csv")


def load_classifier():
    if not MODEL_PATH.exists():
        train()

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    classifier = joblib.load(MODEL_PATH)
    return embedder, classifier


EMBEDDER, CLASSIFIER = load_classifier()


def get_model_description() -> str:
    if not METRICS_PATH.exists():
        return "Best available classifier"

    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics.get("best_classifier", "Best available classifier").replace("_", " ").title()


def classify_feedback(feedback: str):
    feedback = feedback.strip()
    if not feedback:
        return {"Enter feedback text first.": 1.0}

    embedding = EMBEDDER.encode([feedback], normalize_embeddings=True)
    probabilities = CLASSIFIER.predict_proba(embedding)[0]
    return {
        label: float(probability)
        for label, probability in zip(CLASSIFIER.classes_, probabilities)
    }


examples = [
    ["The teacher explains concepts clearly and gives useful examples."],
    ["The library has limited books and the staff is not helpful."],
    ["The lab work is okay but the equipment should be updated."],
]

with gr.Blocks(title="Student Feedback Sentiment Classifier") as demo:
    gr.Markdown("# Student Feedback Sentiment Classifier")
    gr.Markdown(
        "This demo uses sentence embeddings from all-MiniLM-L6-v2 and a "
        f"{get_model_description()} classifier trained on student feedback comments."
    )

    feedback = gr.Textbox(
        label="Student feedback",
        lines=5,
        placeholder="Type a student feedback comment...",
    )
    output = gr.Label(label="Predicted sentiment", num_top_classes=3)
    classify_button = gr.Button("Classify")

    classify_button.click(classify_feedback, inputs=feedback, outputs=output)
    feedback.submit(classify_feedback, inputs=feedback, outputs=output)

    gr.Examples(examples=examples, inputs=feedback)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        gr.Markdown(f"Dataset rows: **{len(df)}**")


if __name__ == "__main__":
    demo.launch()
