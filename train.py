from pathlib import Path
import json

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/student_feedback_sentiment.csv")
MODEL_DIR = Path("models")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42


def build_classifiers() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "linear_svm": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_classifier(classifier, test_embeddings, y_test) -> dict:
    predictions = classifier.predict(test_embeddings)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def train() -> dict:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "sentiment"])

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["sentiment"].tolist(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    train_embeddings = embedder.encode(
        x_train,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    test_embeddings = embedder.encode(
        x_test,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    classifiers = build_classifiers()
    results = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, classifier in classifiers.items():
        print(f"Training {name}...")
        classifier.fit(train_embeddings, y_train)
        results[name] = evaluate_classifier(classifier, test_embeddings, y_test)
        joblib.dump(classifier, MODEL_DIR / f"{name}.joblib")

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["macro_f1"],
            results[name]["accuracy"],
        ),
    )
    best_classifier = classifiers[best_model_name]

    metrics = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "classifiers": list(classifiers.keys()),
        "best_classifier": best_model_name,
        "rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "results": results,
    }

    joblib.dump(best_classifier, MODEL_DIR / "sentiment_classifier.joblib")
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
