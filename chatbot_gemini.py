import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from rapidfuzz import process, fuzz
from spellchecker import SpellChecker

# Set your Gemini API Key securely
os.environ['GOOGLE_API_KEY'] = "AIzaSyDBF3aYLCk94NLOU-ZWhgs3nkoPYBx0h48"
os.environ['GOOGLE_CLOUD_PROJECT'] = "821176173154"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

st.set_page_config(page_title="AMS Chatbot", layout="wide")
st.title("AMS Chatbot")

# Custom CSS to mimic ChatGPT UI
st.markdown("""
    <style>
    .chat-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        max-height: 70vh;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    .message {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        text-align: right;
    }
    .bot-message {
        background-color: #e9ecef;
        color: black;
    }
    .stTextArea textarea {
        height: 100px !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_student_data():
    try:
        df = pd.read_csv('students.csv')
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        df.dropna(subset=['student_id', 'first_name', 'last_name'], inplace=True)
        cols = ['student_id', 'first_name', 'middle_name', 'last_name', 'gender', 'group', 'semester', 'year', 'academic year', 'email']
        for col in cols:
            df[col] = df[col].fillna('').astype(str)
        df['combined'] = df[cols].agg(' '.join, axis=1)
        return df
    except:
        return pd.DataFrame()

student_df = load_student_data()

@st.cache_data
def load_pdf_text(pdf_path):
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except:
        return ""

pdf_texts = [load_pdf_text(path) for path in ["THY_Chantha_CV.pdf", "San_Kimheang_CV.pdf", "ROEUN_SOVANDETH.pdf"]]


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

pdf_chunks = []
for text in pdf_texts:
    pdf_chunks.extend(chunk_text(text))

@st.cache_data
def create_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer().fit(texts)
    tfidf_matrix = vectorizer.transform(texts)
    return vectorizer, tfidf_matrix

combined_texts = list(student_df['combined']) + pdf_chunks
vectorizer, tfidf_matrix = create_tfidf_matrix(combined_texts)


def find_relevant_texts(query, vectorizer, tfidf_matrix, texts, top_n=5, similarity_threshold=0.1):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    if not filtered_indices:
        return []
    filtered_similarities = similarities[filtered_indices]
    top_indices = np.array(filtered_indices)[filtered_similarities.argsort()[-top_n:][::-1]]
    return [texts[i] for i in top_indices]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        msg_class = "user-message" if message["role"] == "user" else "bot-message"
        st.markdown(f'<div class="message {msg_class}">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

prompt = st.text_area("Type your message...", height=100, key="prompt_input")

spell = SpellChecker()

@st.cache_data
def load_instructor_data():
    try:
        df = pd.read_csv('instructors_info.csv')
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        return df
    except:
        return pd.DataFrame()

def correct_spelling(text: str) -> str:
    words = text.split()
    corrected_words = [spell.correction(word) if spell.unknown([word]) else word for word in words]
    return " ".join(corrected_words)

@st.cache_data
def log_evaluation(query, response):
    if 'eval_log' not in st.session_state:
        st.session_state.eval_log = []
    st.session_state.eval_log.append({'query': query, 'response': response})

if st.button("Send"):
    if prompt.strip():
        name = prompt.strip()
        relevant_texts = find_relevant_texts(name, vectorizer, tfidf_matrix, combined_texts)
        context = "\n".join(relevant_texts) if relevant_texts else "No relevant data found."
        final_prompt = f"Context:\n{context}\n\nQuestion:\n{name}"
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(final_prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error: {e}"

        st.session_state.chat_history.append({"role": "user", "content": name})
        st.session_state.chat_history.append({"role": "bot", "content": answer})
        st.markdown(answer)
        log_evaluation(name, answer)

if st.checkbox("Show Evaluation Log"):
    st.write(st.session_state.eval_log)

st.markdown("""
### Feedback Loop
- Collect user feedback on answers.
- Use feedback to improve data quality, embeddings, and prompt design iteratively.
""")
