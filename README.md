# Alzheimer's Disease Classification using Machine Learning

A machine learning project that compares multiple supervised learning algorithms for Alzheimer's disease classification using clinical, lifestyle, and cognitive assessment data.

---

## Motivation

This project was inspired by a personal experience with Alzheimer's disease and my growing interest in applying machine learning to healthcare challenges. The primary goal was to implement, compare, and evaluate different classification algorithms on a real-world medical dataset.

---

## Project Overview

The objective of this project is to compare the performance of four supervised machine learning algorithms for binary classification of Alzheimer's disease.

The workflow includes:

- Data preprocessing
- Feature selection
- Model training
- Model evaluation
- Cross-validation
- Hyperparameter tuning
- Feature importance analysis

---

## Dataset

The dataset contains demographic, lifestyle, clinical, and cognitive assessment information.

### Target Variable

- **Diagnosis**
  - 0 → Healthy
  - 1 → Dementia

### Removed Features

The following variables were removed because they do not contribute to prediction:

- PatientID
- DoctorInCharge

---

## Machine Learning Models

The following models were implemented and compared:

- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Random Forest

---

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- Area Under the Curve (AUC)
- 5-Fold Cross Validation

Random Forest hyperparameters were optimized using GridSearchCV.

---

## Results

| Model | Accuracy | AUC |
|-------|---------:|----:|
| Logistic Regression | 81.63% | 0.885 |
| Naive Bayes | 77.21% | 0.850 |
| Decision Tree | 90.00% | 0.894 |
| **Random Forest** | **94.88%** | **0.937** |

Among all evaluated models, **Random Forest achieved the best overall performance**.

---

## Feature Importance

Random Forest identified the following variables as the most influential predictors:

- Functional Assessment
- ADL
- MMSE
- Memory Complaints
- Behavioral Problems

---

## Presentation

The project presentation slides are available here:

- [Presentation Slides](presentation/presentation.pdf)
---


## Repository Structure

```text
alzheimers-disease-classification/

├── data/
├── figures/
├── presentation/
├── src/
│   ├── main.py
│   ├── data_processing.py
│   ├── model_training.py
│   └── visualization.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/alzheimers-disease-classification.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the project

```bash
python -m src.main
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Future Improvements

Potential extensions include:

- Support Vector Machines (SVM)
- XGBoost
- LightGBM
- Feature Selection techniques
- External dataset validation
- Explainable AI (SHAP / LIME)

---

## Author

**Melika Rahimpour Rahimpour**

Computer Science Graduate

Interested in Machine Learning, Medical AI, and Data Science.