# Machine Learning-Based Email Phishing Detection

A Python tool that trains multiple machine learning classifiers to detect phishing emails and then uses the best model to classify new emails.

## Project Overview
This project compares six models over repeated runs, then selects and saves the best-performing model and TF-IDF vectorizer for future classification.

### Models compared
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

## Features
- Load and clean phishing + legitimate datasets
- Train and evaluate multiple models (accuracy, precision, recall, F1)
- Compare results over multiple runs
- Save/load best model with `joblib`
- Classify new emails (manual input or from text file)
- Show confidence scores when available

## Repository Structure
```text
.
├── ML-based_phishing_detector.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── phishing_emails.csv              # optional to track
├── legitimate_emails.csv            # optional to track
├── best_model.joblib                # generated artifact (ignored by default)
├── vectorizer.joblib                # generated artifact (ignored by default)
└── model_info.txt                   # generated artifact (ignored by default)
```

## Requirements
- Python 3.9+
- pip

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the tool
```bash
python ML-based_phishing_detector.py
```

## Data Format
Expected CSV columns:
- `email_text` (string)
- `label` (int): `1` = phishing, `0` = legitimate

## Typical Workflow
1. Prepare datasets (`phishing_emails.csv`, `legitimate_emails.csv`)
2. Run training/comparison from the menu
3. Save best model and vectorizer
4. Classify new emails using saved artifacts

## Notes
- Model artifacts (`*.joblib`) are ignored by default to keep repository lightweight.
- If you want to publish trained models, remove those patterns from `.gitignore`.
- If datasets are private/sensitive, keep them out of public repositories.

## Suggested GitHub Steps
```bash
git init
git add .
git commit -m "Initial commit: ML phishing detector"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Acknowledgment
This README is aligned with your uploaded project report and implementation/results descriptions.
