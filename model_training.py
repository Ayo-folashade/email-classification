# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv("spam-text-message-classification/SPAM text message 20170820 - Data.csv", encoding='latin1')

# Extract the email content and target labels
X = df['Message'].tolist()
y = df['Category'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Handle class imbalance by undersampling the majority class (ham)
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_features, y_train)

# Create individual classifiers
logistic_regression = LogisticRegression(max_iter=1000)
naive_bayes = MultinomialNB()

# Create the VotingClassifier
voting_classifier = VotingClassifier(
    estimators=[('lr', logistic_regression), ('nb', naive_bayes)],
    voting='soft'
)

# Train the ensemble model
voting_classifier.fit(X_train_resampled, y_train_resampled)

# Evaluate the ensemble model
y_pred = voting_classifier.predict(X_test_features)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")
