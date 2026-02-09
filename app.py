import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="News Credibility Checker")

# Load saved model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

st.title("News Credibility Checker")
st.write("Enter a news headline or article text to evaluate its credibility.")

user_input = st.text_area("News Text")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        confidence = round(max(probability) * 100, 2)

        if prediction == 1:
            st.error(f"This content is likely fake. Confidence: {confidence}%")
        else:
            st.success(f"This content appears to be real. Confidence: {confidence}%")
