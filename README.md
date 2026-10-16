# OP_SENTINEL_AI: Phishing Email Classifier

## 🛡️ Project Overview
`OP_SENTINEL_AI` is a machine learning-based security tool designed to identify phishing emails. By utilizing Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier, the system analyzes the linguistic patterns of an email to determine if it is a legitimate communication (HAM) or a fraudulent attempt (PHISH).

## ⚙️ Technical Architecture
The project is implemented as a modular pipeline to ensure scalability and maintainability:

1. **Data Layer (`emails.csv`)**: Contains a labeled dataset of emails used for supervised learning.
2. **Preprocessing Layer (`data_processor.py`)**: 
   - **Tokenization**: Breaking down text into individual words.
   - **Stop-word Removal**: Filtering out common English words (e.g., "the", "is") that do not contribute to sentiment.
   - **TF-IDF Vectorization**: Converting text into numerical vectors using Term Frequency-Inverse Document Frequency to highlight "suspicious" keywords.
3. **Training Layer (`model_trainer.py`)**: 
   - Uses a **Multinomial Naive Bayes** algorithm, which is highly efficient for high-dimensional text data.
   - Implements an 80/20 train-test split for validation.
4. **Inference Layer (`predict.py`)**: A real-time classifier that provides a probability-based threat score for new emails.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Pandas
- Scikit-learn

### Installation
```bash
pip install pandas scikit-learn
```

### Execution Flow
1. **Train the AI**:
   ```bash
   python model_trainer.py
   ```
   *This creates `vectorizer.pkl` and `phishing_model.pkl`.*

2. **Analyze Emails**:
   ```bash
   python predict.py
   ```

## 📊 Performance Metrics
The model's effectiveness is measured using a Classification Report including:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The ability of the classifier not to label as phishing a legitimate email.
- **Recall**: The ability of the classifier to find all the phishing emails.

## 🛡️ Security Implications
This tool serves as a first line of defense in an organization's security stack, reducing the human error factor in social engineering attacks.
