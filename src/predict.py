import pickle

def predict_email(model_file, vectorizer_file, email_text):
    # Load the trained model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load the vectorizer
    with open(vectorizer_file, "rb") as f:
        _, _, vectorizer = pickle.load(f)

    # Vectorize the input email (keep as sparse)
    email_vector = vectorizer.transform([email_text])

    # Make a prediction
    prediction = model.predict(email_vector)
    return "Phishing" if prediction[0] == 1 else "Not Phishing"

if __name__ == "__main__":
    print("Phishing Email Detector (type 'exit' to quit)\n")
    while True:
        email_text = input("Enter email text: ")
        if email_text.lower() == "exit":
            print("Exiting...")
            break
        result = predict_email(
            "models/phishing_detector.pkl",
            "data/preprocessed_data.pkl",
            email_text
        )
        print(f"Prediction: {result}\n")
