import streamlit as st
import pandas as pd
import nltk
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Smart FAQ Chatbot")
st.write("Ask me anything from the FAQ database!")

# Load data
data = pd.read_csv("faq.csv")

questions = data["Question"].tolist()
answers = data["Answer"].tolist()

# Vectorization
vectorizer = CountVectorizer()
question_vectors = vectorizer.fit_transform(questions)

user_input = st.text_input("💬 Type your question")

if user_input:

    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()
    best_score = similarity.max()

    st.markdown("### 🤖 Bot is thinking...")

    # ⏳ Typing animation
    with st.empty():
        typed_text = ""
        response = answers[best_match_index]

        # ❗ SMART FALLBACK (Option 7)
        if best_score < 0.3:
            typed_text = "Sorry 😕 I don't understand that question. Try asking something from the FAQ."
            st.warning(typed_text)
        else:
            for char in response:
                typed_text += char
                time.sleep(0.03)  # typing speed
                st.markdown(typed_text)