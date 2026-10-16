import pickle
from data_processor import DataProcessor

def predict_email():
    print("\n" + "="*40)
    print("   OP_SENTINEL_AI: PHISHING DETECTOR")
    print("="*40)

    try:
        # Load the saved vectorizer and model
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('phishing_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("[!] Error: Model files not found. Please run model_trainer.py first!")
        return

    while True:
        user_input = input("\nEnter email text to analyze (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        # Preprocess input
        text_vector = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]

        confidence = probability[prediction] * 100

        if prediction == 1:
            print(f"RESULT: [!] PHISHING DETECTED ({confidence:.2f}% confidence)")
        else:
            print(f"RESULT: [✓] LEGITIMATE EMAIL ({confidence:.2f}% confidence)")

if __name__ == "__main__":
    predict_email()
