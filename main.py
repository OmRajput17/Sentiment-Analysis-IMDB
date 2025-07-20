# Importing all the libraries

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Loading the word index

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

# Loading the model

model = load_model('IMDB_Classification.h5')

# Helper Functions

def decode_review(encoded_review):
    return " ".join(reverse_word_index.get(i - 3, '?') for i in encoded_review)

# Function to preprocess the user input
def preprocess_data(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review


# prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_data(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# StreamLit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to Classify it as +ve or -ve")

# User Input

user_input = st.text_area('Movie Review')

if st.button('Classify'):
  # Make the prediction
    sentiment, prediction = predict_sentiment(user_input)
    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediciton Score: {prediction}")
else:
    st.write('Please enter a movie review.')

