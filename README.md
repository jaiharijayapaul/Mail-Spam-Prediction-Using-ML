# 🛡️ Mail Spam Guard - Machine Learning Spam Detection

A high-performance machine learning system and interactive web dashboard for real-time email spam detection and analysis.

This repository features an end-to-end classification pipeline comparing **Multinomial Naive Bayes** and **Logistic Regression** using term frequency representations (**CountVectorizer** and **TF-IDF Vectorizer**). It achieves up to **98.12% accuracy** in distinguishing normal emails (*ham*) from unwanted messages (*spam*).

---

## 🚀 Key Features

* **Dataset Insights**: Interactive class balance visualizations and key word frequency extraction.
* **Custom Text Preprocessing**: Strips punctuation, converts to lowercase, and filters out common stop words using a custom NLP pipeline.
* **Algorithm Comparison**: Side-by-side performance evaluation metrics (Accuracy, Precision, Recall, F1-Score) and interactive Confusion Matrices.
* **Interactive Live Prediction**: A highly polished playground interface with simulated analysis loading states, prediction confidence probabilities, and preprocessed vocabulary inspection.
* **Premium Design**: Built using custom dark/light theme options, card-based metrics, and interactive charts (Plotly).

---

## 📊 Model Evaluation Summary

The models were trained on 80% of the dataset and evaluated on a stratified 20% test split:

| Model Configuration | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Bayes (CountVectorizer)** 🌟 | **98.12%** | **97.06%** | **88.59%** | **0.9263** |
| **Logistic Regression (CountVectorizer)** | **98.03%** | **100.00%** | **85.23%** | **0.9203** |
| **Logistic Regression (TF-IDF)** | **96.14%** | **100.00%** | **71.14%** | **0.8314** |
| **Naive Bayes (TF-IDF)** | **96.05%** | **100.00%** | **70.47%** | **0.8268** |

### Insights:
* **Feature Representation**: Word counts (`CountVectorizer`) proved superior to normalized frequencies (`TF-IDF`) for short text messages, achieving a ~18% higher spam detection recall.
* **Model Choices**: Naive Bayes offers the best overall recall (fewer missed spams), while Logistic Regression offers perfect precision (zero false alarms on normal emails).

---

## 📁 Repository Structure

```text
├── dataset/
│   └── mail_data.csv          # Raw dataset containing 5,572 rows
├── app.py                     # Streamlit web application dashboard code
├── spam_detection_analysis.ipynb # Jupyter notebook showing the step-by-step training pipeline
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mail-spam-detection.git
   cd mail-spam-detection
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Run the analysis notebook:**
   ```bash
   jupyter notebook spam_detection_analysis.ipynb
   ```

---

## ⚙️ How it Works

1. **Preprocessing**: Cleans raw text into tokens by converting it to lowercase, stripping punctuation, and removing standard English stopwords.
2. **Vectorization**: Transforms cleaned text tokens into numerical vectors using a fitted `CountVectorizer`.
3. **Classification**: Computes spam probability and predicts final class tags (`HAM - SAFE` or `SPAM - UNWANTED`) using the trained classifier model.
