import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('fake_news_model')  # Adjust this to your model path
tokenizer = BertTokenizer.from_pretrained('fake_news_model')  # Adjust this to your tokenizer path

# Streamlit App UI
st.title("Fake News Classification")
st.write("Enter the news text below to classify it as real or fake.")

# Text input box for the news article
news_text = st.text_area("News Text")

# Function to predict fake or real news
def predict_news(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Perform inference with the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction

# Button to classify the news
if st.button("Classify News"):
    if news_text:
        result = predict_news(news_text)
        
        # Display results
        if result == 0:
            st.success("The news is *REAL*")
        else:
            st.error("The news is *FAKE*")
    else:
        st.warning("Please enter a news article to classify.")