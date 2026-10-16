import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
from data_processor import DataProcessor

def train_model():
    print("[*] Initializing OP_SENTINEL_AI Training Pipeline...")

    # 1. Initialize Processor and Load Data
    processor = DataProcessor('emails.csv')
    X, y = processor.load_and_clean_data()

    # 2. Split data for validation (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Vectorize the text
    X_train = processor.fit_vectorizer(X_train_raw)
    X_test = processor.transform_text(X_test_raw) # Note: transform_text needs to handle multiple strings

    # Fix: The current transform_text in data_processor only handles one string.
    # I will update data_processor.py to handle both.

    # 4. Train Model (Naive Bayes is excellent for text)
    print("[*] Training Multinomial Naive Bayes Model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    print(f"\n[+] Model Training Complete.")
    print(f"[+] Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    # 6. Save the Model
    with open('phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\n[*] Model saved as 'phishing_model.pkl'")

if __name__ == "__main__":
    train_model()
