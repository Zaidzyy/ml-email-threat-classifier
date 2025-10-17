# **ML Email Threat Classifier**

This project builds a machine learning-based system to detect phishing emails. It uses Natural Language Processing (NLP) for text vectorization and a Random Forest classifier to classify emails as phishing or legitimate. The goal is to help users and organizations automatically identify phishing attempts based on email content. <br> Author-Mohammed Zaid

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Setup and Usage](#setup-and-usage)
  - [Step 1: Install Dependencies](#step-1-install-dependencies)
  - [Step 2: Preprocess Data](#step-2-preprocess-data)
  - [Step 3: Train the Model](#step-3-train-the-model)
  - [Step 4: Make Predictions](#step-4-make-predictions)
  - [Step 5: Run the Web Interface](#step-5-run-the-web-interface)
- [Demo Image](#demo-image)
- [How It Works](#how-it-works)

---

## **Project Overview**

This project detects phishing emails using machine learning. It involves several steps, including preprocessing raw email data, training a model using a labeled dataset, and using that model to predict whether new emails are phishing or legitimate.

---

## **Features**
- **Email Classification**: Classify emails into "Phishing" or "Not Phishing" based on their content.
- **Text Preprocessing**: The raw email text is converted into a feature vector using TF-IDF (Term Frequency - Inverse Document Frequency).
- **Model Training**: Train a machine learning model using various algorithms such as Random Forest, Logistic Regression, etc.
- **Prediction**: Predict whether a new email is phishing based on the trained model.
- **Interactive Web Interface**: Test email content via a browser-based interface using Flask.

---

## **Technologies Used**
- **Python**: Programming language.
- **Scikit-learn**: Library for machine learning algorithms.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Library for numerical computing.
- **Flask**: used to deploy the model as a web service.
- **Pickle**: For saving and loading the trained model.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/ml-email-threat-classifier.git
cd ml-email-threat-classifier


```

## **Setup and Usage**

### **Step 1: Install Dependencies**
Make sure all required libraries are installed:
```bash
pip install -r requirements.txt

```

### **Step 2: Preprocess Data**
Run the `preprocess.py` script to preprocess the raw data and save it as a pickle file:
```bash
python src/preprocess.py
```
This script:
- Loads the raw email dataset (`phishing_emails.csv`).
- Converts the email text into a numerical format using TF-IDF vectorization.
- Saves the processed data to `preprocessed_data.pkl`.

### **Step 3: Train the Model**
Train the machine learning model by running the `train.py` script:
```bash
python src/train.py
```
This script:
- Loads the preprocessed data.
- Splits it into training and testing sets.
- Trains a Random Forest classifier (or other ML models) using the training data.
- Saves the trained model to `phishing_detector.pkl`.

**Note**: To try a different model, replace the `RandomForestClassifier` in `train.py` with another algorithm, like `LogisticRegression`.

### **Step 4: Make Predictions**
Predict phishing emails using the `predict.py` script:
```bash
python src/predict.py
```
This script:
- Loads the trained model (`phishing_detector.pkl`) and vectorizer.
- Takes input email content from the user and predicts whether it’s phishing or legitimate.

### **Step 4: Run the Web Interface**
To run the Web Server using the `app.py` script:
```bash
python src/app.py
```
Open a browser and go to http://127.0.0.1:5000.

Paste email content in the input box and click Check Email Content.

The interface will return "Phishing" or "Not Phishing".

app.py uses the trained model (phishing_detector.pkl) and vectorizer (preprocessed_data.pkl) to provide real-time predictions.

---

## **Demo Image**

demo_image.jpg

---

## **How It Works**

### **1. Preprocessing**
- Converts raw email content into numerical feature vectors using TF-IDF vectorization. This calculates the importance of each word in the email relative to other emails.

### **2. Model Training**
- Trains a Random Forest model to classify emails as phishing or not based on text patterns.

### **3. Prediction**
- For a new email, the model predicts whether it is phishing or legitimate using the patterns it learned during training.

### **4. Web Interface**
- app.py provides an interactive way to test emails in a browser.
---


