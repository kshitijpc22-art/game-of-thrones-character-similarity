import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.title("📧 Spam Email Classifier")

email_input = st.text_area("Enter Email Content", height=200)

if st.button("Classify"):
    if email_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed = transform_text(email_input)
        vectorized = tfidf.transform([transformed])
        proba = model.predict_proba(vectorized)[0]
        st.write(f"Spam probability: {proba[1]*100:.1f}%")
        if proba[1] >= 0.3:
            st.error(f"🚨 SPAM! (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ Not Spam! (Confidence: {proba[0]*100:.1f}%)")