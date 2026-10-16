import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    def load_and_clean_data(self):
        """Loads CSV and separates text from labels."""
        df = pd.read_csv(self.file_path)
        X = df['text']
        y = df['label'].map({'phish': 1, 'ham': 0})
        return X, y

    def fit_vectorizer(self, X):
        """Turns text into numerical TF-IDF vectors."""
        X_vectorized = self.vectorizer.fit_transform(X)
        # Save vectorizer for use in prediction
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        return X_vectorized

    def transform_text(self, text):
        """Transforms one or more strings into vectors."""
        if isinstance(text, str):
            return self.vectorizer.transform([text])
        return self.vectorizer.transform(text)

if __name__ == "__main__":
    print("[*] DataProcessor module ready.")
