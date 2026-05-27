from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data_raw/finalDataset0.2.xlsx")
OUTPUT_PATH = Path("data/student_feedback_sentiment.csv")

SENTIMENT_LABELS = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}


def normalize_category(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "_")
    return {
        "coursecontent": "course_content",
        "library_facilities": "library_facilities",
    }.get(cleaned, cleaned)


def build_long_dataset(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(raw_path)
    records = []

    columns = list(df.columns)
    for label_col, text_col in zip(columns[0::2], columns[1::2]):
        category = normalize_category(str(label_col))

        for _, row in df[[label_col, text_col]].iterrows():
            raw_label = row[label_col]
            text = row[text_col]

            if pd.isna(raw_label) or pd.isna(text):
                continue

            try:
                numeric_label = int(raw_label)
            except (TypeError, ValueError):
                continue

            if numeric_label not in SENTIMENT_LABELS:
                continue

            text = str(text).strip()
            if not text:
                continue

            records.append(
                {
                    "text": text,
                    "category": category,
                    "sentiment_score": numeric_label,
                    "sentiment": SENTIMENT_LABELS[numeric_label],
                }
            )

    long_df = pd.DataFrame(records)
    long_df = long_df.drop_duplicates(subset=["text", "category", "sentiment"])
    return long_df.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_long_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(dataset)} rows to {OUTPUT_PATH}")
    print(dataset["sentiment"].value_counts().to_string())
    print(dataset["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
