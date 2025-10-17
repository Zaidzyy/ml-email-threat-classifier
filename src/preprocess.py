import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)

    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("Input CSV must have 'text' and 'label' columns.")

    # Drop rows where text or label is missing
    data = data.dropna(subset=['text','label'])

    # Ensure labels are integers
    data['label'] = data['label'].astype(int)

    # TF-IDF vectorizer (sparse matrix)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['text'])

    y = data['label'].values

    # Save preprocessed data
    with open(output_file, "wb") as f:
        pickle.dump((X, y, vectorizer), f)

    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data("data/phishing_emails.csv", "data/preprocessed_data.pkl")
