# Credit Card Fraud Detection with SMOTE

A machine learning pipeline for detecting credit card fraud on a highly imbalanced dataset, using **SMOTE** (Synthetic Minority Oversampling Technique) combined with **Random Undersampling** to address class imbalance. Two classifiers are trained and evaluated: Logistic Regression and a Neural Network built with TensorFlow/Keras.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Results](#results)
- [Technologies](#technologies)

---

## Overview

Credit card fraud datasets are extremely imbalanced — fraudulent transactions represent a tiny fraction of all records. Training a model on raw imbalanced data leads to poor recall on the minority class (fraud). This project tackles that problem with a hybrid resampling strategy:

- **SMOTE** — generates synthetic samples for the minority class (fraud)
- **Random Undersampling** — reduces the majority class (legitimate transactions)

The combined pipeline produces a balanced training set, which is then used to train both a Logistic Regression model and a deep learning model.

---

## Dataset

The notebook uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`).

| Property | Details |
|---|---|
| Features | 30 (V1–V28 are PCA-transformed, plus `Time` and `Amount`) |
| Target | `Class` — `0` = legitimate, `1` = fraud |
| Imbalance | ~0.17% of transactions are fraudulent |

> **Note:** Download the dataset from Kaggle and place it at `/content/creditcard.csv` (or update the path in the notebook).

---

## Project Structure

```
├── SMOTE.ipynb       # Main notebook
└── README.md
```

---

## Installation

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn yellowbrick tensorflow
```

Or with a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
yellowbrick
tensorflow
```

---

## Usage

1. Clone this repository and navigate into it.
2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `/content/` (or update the path in Cell 1).
3. Open and run the notebook:

```bash
jupyter notebook SMOTE.ipynb
```

---

## Pipeline

### 1. Exploratory Data Analysis (EDA)
- Class distribution bar chart — visualizes the severe imbalance
- Correlation heatmap across all features
- Distribution of `Time` and boxplot of `Amount`
- Scatter plot of fraud transactions by time and amount
- Feature correlation ranking with respect to the `Class` target

### 2. Preprocessing
- Drop null values and duplicate rows
- Standard scaling of all features with `StandardScaler`

### 3. Resampling (Imbalanced-learn Pipeline)
```
SMOTE → RandomUnderSampler
```
- SMOTE oversamples the fraud class synthetically
- Random undersampling reduces the majority class
- Before/after scatter plots compare the class distributions visually

### 4. Model Training & Evaluation

**Logistic Regression**
- Trained on the resampled dataset
- Evaluated with a classification report and confusion matrix
- Visualized using `yellowbrick`'s `ClassificationReport`

**Neural Network (TensorFlow/Keras)**
```
Input → Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)
```
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Trained for 10 epochs with validation on the test set

---

## Results

| Metric | Logistic Regression | Neural Network |
|---|---|---|
| Evaluated on | Original imbalanced test set | Original imbalanced test set |
| Key focus | Precision / Recall on fraud class | Test accuracy |

> Exact metric values will appear in the notebook output after running all cells.

---

## Technologies

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `scikit-learn` | Preprocessing, splitting, Logistic Regression, metrics |
| `imbalanced-learn` | SMOTE, RandomUnderSampler, Pipeline |
| `matplotlib` / `seaborn` | Visualization |
| `yellowbrick` | Visual classification report |
| `TensorFlow / Keras` | Deep learning model |
