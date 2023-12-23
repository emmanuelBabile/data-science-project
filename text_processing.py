import re
import streamlit as st
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

def extract_email(uploaded_text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, uploaded_text, re.IGNORECASE)
    if emails:
        st.write("Here are emails from the file:")
        return emails
    else:
        st.write("There are no email in the text file.")
        return None

def text_lenght(uploaded_text):
    st.write("The length of the text is:")
    return len(uploaded_text)

def verify_color(uploaded_text):
    pattern = re.compile(r'#([a-fA-F0-9]{3}){1,2}\b')

    color_codes = re.findall(pattern, uploaded_text)

    if color_codes:
        st.write("Color codes found in the text:")
        for color_code in color_codes:
            st.write(f"#{color_code} is a valid color code.")
    else:
        st.write("There are no color codes in the text file.")

    return color_codes

def detect_language(uploaded_text):
    try:
        language = detect(uploaded_text)
        return f"The detected language is: {language}"
    except Exception as e:
        return f"Language detection failed with error: {str(e)}"

def clean_text(uploaded_text):
    # Remove special characters and punctuation
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', uploaded_text)
    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def remove_stopwords(uploaded_text):
    stop_words = set(stopwords.words('french'))  # Adapt according to language
    words = word_tokenize(uploaded_text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def analyze_sentiment(uploaded_text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(uploaded_text)

    # Retrieve polarity score
    compound_score = sentiment_score['compound']

    # Interpret polarity in terms of feelings
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment
