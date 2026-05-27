---
title: Student Feedback Sentiment Classifier
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Student Feedback Sentiment Classifier Using Embeddings

## Project Links

- GitHub repository: https://github.com/hanneslindqvist7265/Information-retrieval-assignment-1-
- Hugging Face dataset: https://huggingface.co/datasets/Hannes111/student_feedback_sentiment
- Hugging Face demo: https://huggingface.co/spaces/Hannes111/student-feedback-sentiment-classifier 


## Project Goal

This project trains a classifier for a specific education-domain problem: predicting the sentiment of student feedback comments.

The classifier uses sentence embeddings to represent student comments as numerical vectors, then trains a supervised model to classify each comment as:

- negative
- neutral
- positive

## Problem Definition

Universities collect feedback about teaching, course content, examinations, lab work, library facilities, and extracurricular activities. Manually reading and sorting this feedback is slow. This project helps identify whether feedback is positive, neutral, or negative so that staff can quickly understand areas that need attention.

## Challenge

Student feedback is often short, informal, and inconsistent. A keyword-only approach can miss the meaning of comments such as "not bad" or "needs improvement." Embeddings help because they capture semantic meaning beyond exact word matching.

## Dataset

The original dataset is the Kaggle student feedback spreadsheet:

`brarajit18/student-feedback-dataset`

The raw file is included in:

`data_raw/finalDataset0.2.xlsx`

The cleaned project dataset is:

`data/student_feedback_sentiment.csv`

The cleaned dataset reshapes the spreadsheet into one row per feedback comment with these columns:

- `text`: the student feedback comment
- `category`: the feedback area, such as teaching or labwork
- `sentiment_score`: original numeric label, -1, 0, or 1
- `sentiment`: readable class label, negative, neutral, or positive

## Embeddings

This project uses:

`sentence-transformers/all-MiniLM-L6-v2`

Each feedback comment is converted into a sentence embedding. The embeddings are then used as input features for several supervised classifiers.

## Model

The model pipeline is:

1. Load cleaned student feedback dataset
2. Convert feedback text into sentence embeddings
3. Train Logistic Regression, Linear SVM, and Random Forest classifiers
4. Evaluate all classifiers on the same held-out test split
5. Save all trained classifiers and select the best model by macro F1-score
6. Save the selected classifier to `models/sentiment_classifier.joblib`

The training script saves:

- `models/logistic_regression.joblib`
- `models/linear_svm.joblib`
- `models/random_forest.joblib`
- `models/sentiment_classifier.joblib`
- `models/metrics.json`

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python src/prepare_data.py
```

Train the model:

```bash
python train.py
```

Launch the demo:

```bash
python app.py
```

## Hugging Face Space

This repository is ready to upload as a Hugging Face Space.

Recommended Space settings:

- SDK: Gradio
- App file: `app.py`
- Python version: default

When the Space starts, `app.py` loads the trained model. If the saved model does not exist yet, it trains the classifier automatically from the included dataset.

## Lab Requirement Checklist

| Requirement | How this project satisfies it |
|---|---|
| Identify an issue in a specific domain | Student feedback sentiment classification in education |
| Define the challenge | Manual review is slow and short comments are semantically varied |
| Specify embeddings | Uses `sentence-transformers/all-MiniLM-L6-v2` |
| Hosted on Hugging Face | Designed as a Gradio Hugging Face Space |
| Include a custom dataset | Includes cleaned `data/student_feedback_sentiment.csv` |
| Include a trained model | `train.py` saves all classifiers and selects the best model |
| Provide a working demo space | `app.py` provides the interactive Gradio demo |

## Future Improvements

- Add more student feedback examples to balance classes
- Train separate classifiers for each feedback category
- Add confidence thresholds for uncertain predictions
- Build a dashboard showing sentiment by category
