# import required libraries
import pickle

# Load the pickled objects
with open('voting_classifier.pkl', 'rb') as file:
    voting_classifier = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Sample dataset
sample_messages = [
    "Earn effortless cash from the comfort of your home! Simply enroll in our exceptional money-making program.",
    "URGENT: You've been awarded a lavish vacation for two! Claim your prize immediately.",
    "Boost your fortune with our confidential investment strategy. Limited-time opportunity!",
    "Experience a FREE trial of our groundbreaking weight loss pill. Shed pounds within days!",
    "Congratulations! You've been chosen as the lucky recipient of a brand new car.",
    "URGENT: Your bank account has been compromised. Click here to ensure its security.",
]

# Transform the sample dataset
sample_features = vectorizer.transform(sample_messages)

# Perform classification on the sample dataset
sample_predictions = voting_classifier.predict(sample_features)

# Display the results
for message, prediction in zip(sample_messages, sample_predictions):
    print(f"Message: {message}")
    print(f"Prediction: {prediction}\n")
