# Import necessary libraries
import streamlit as st
import pickle

# Load the pickled objects
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('voting_classifier.pkl', 'rb') as file:
    voting_classifier = pickle.load(file)


# Deploy the application on Streamlit
def classify_spam(message):
    # Preprocess the input message using vectorizer
    processed_message = vectorizer.transform([message])
    # Generate predictions using the voting classifier
    prediction = voting_classifier.predict(processed_message)

    return prediction[0]

# Set the page title and description
def main():
    st.title("Email Classification")
    st.markdown("This web application allows you to classify emails as either spam or not spam.")

    # Text input for users to enter a message
    user_input = st.text_input("Enter the email content below:")

    # Button for initiating the prediction process
    if st.button("Classify"):
        if user_input:
            # Classify the message based on the input
            prediction = classify_spam(user_input)

            # Display the prediction result
            if prediction == 'spam':
                st.error("This message has been categorized as spam.")
            else:
                st.success("This message has been categorized as not spam.")


if __name__ == '__main__':
    main()
