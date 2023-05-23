# Email Classification Model
This repository contains a machine learning model for email classification, distinguishing between spam and non-spam emails. The model is trained using a dataset of labeled spam and non-spam emails. It utilizes a combination of logistic regression and Naive Bayes algorithms in an ensemble approach, implemented through a VotingClassifier.

## Features
- Data preprocessing: The model extracts and transforms email content into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) technique.
- Class imbalance handling: To tackle class imbalance, the majority class (non-spam) is undersampled during training.
- Ensemble model: The VotingClassifier combines predictions from logistic regression and Naive Bayes classifiers using a soft voting strategy.
- Evaluation metrics: Model performance is evaluated using accuracy, precision, recall, and F1 score.

## Usage
- Install the required dependencies listed in the [requirements.txt]() file by running 
```
pip install -r requirements.txt.
```
- Run the [streamlit_app.py]() script to launch the Streamlit application locally.
- Access the application in your browser using the provided local URL.
- Enter the email content in the input box and click "Classify" to obtain the prediction of whether the email is spam or not.
- The result will be displayed to show whether the email content is spam or not.

## About
This email classification model aims to enhance email management by effectively identifying and filtering spam emails. It can be integrated into existing email systems or utilized as a standalone solution.
