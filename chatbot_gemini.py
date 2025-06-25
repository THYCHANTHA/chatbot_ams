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
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# Load and clean student data
@st.cache_data
def load_student_data():
    cols = ['student_id', 'first_name', 'middle_name', 'last_name', 'gender', 'group', 'semester', 'year', 'academic year', 'email']
    try:
        df = pd.read_csv('students.csv')
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        df.dropna(subset=['student_id', 'first_name', 'last_name'], inplace=True)
        for col in cols:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].fillna('').astype(str)
        df['combined'] = df[cols].agg(' '.join, axis=1)
        return df
    except FileNotFoundError:
        st.error("students.csv file not found.")
        empty_df = pd.DataFrame(columns=cols)
        empty_df['combined'] = ''
        return empty_df
    except Exception as e:
        st.error(f"Error loading student data: {e}")
        empty_df = pd.DataFrame(columns=cols)
        empty_df['combined'] = ''
        return empty_df

student_df = load_student_data()

# Extract text from PDF
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
    except FileNotFoundError:
        st.error(f"PDF file {pdf_path} not found.")
        return ""
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

pdf_text = load_pdf_text("THY_Chantha_CV.pdf")
pdf_text2 = load_pdf_text("San_Kimheang_CV.pdf")
pdf_text3 = load_pdf_text("ROEUN_SOVANDETH.pdf")

# Split text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

pdf_chunks = chunk_text(pdf_text) + chunk_text(pdf_text2) + chunk_text(pdf_text3)

# Create TF-IDF vectorizer and matrix for student data and pdf chunks
@st.cache_data
def create_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer().fit(texts)
    tfidf_matrix = vectorizer.transform(texts)
    return vectorizer, tfidf_matrix

if 'combined' in student_df.columns and not student_df['combined'].empty:
    combined_texts = list(student_df['combined'])
else:
    combined_texts = []

combined_texts += pdf_chunks

if combined_texts:
    vectorizer, tfidf_matrix = create_tfidf_matrix(combined_texts)
else:
    vectorizer, tfidf_matrix = None, None

def find_relevant_texts(query, vectorizer, tfidf_matrix, texts, top_n=5, similarity_threshold=0.1):
    if vectorizer is None or tfidf_matrix is None:
        return []
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    if not filtered_indices:
        return []
    filtered_similarities = similarities[filtered_indices]
    top_indices = np.array(filtered_indices)[filtered_similarities.argsort()[-top_n:][::-1]]
    return [texts[i] for i in top_indices]

# Prompt input and chat history
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

# Evaluation log
if 'eval_log' not in st.session_state:
    st.session_state.eval_log = []

def log_evaluation(query, response):
    st.session_state.eval_log.append({'query': query, 'response': response})

spell = SpellChecker()

@st.cache_data
def load_instructor_data():
    try:
        df = pd.read_csv('instructors_info.csv')
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        return df
    except FileNotFoundError:
        st.error("instructors_info.csv file not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading instructor data: {e}")
        return pd.DataFrame()

def correct_spelling(text: str) -> str:
    words = text.split()
    corrected_words = [spell.correction(word) if spell.unknown([word]) else word for word in words]
    return " ".join(corrected_words)

def format_student_summary(row):
    name_parts = [str(row['first_name'])]
    middle_name = row.get('middle_name', '')
    if middle_name and str(middle_name).lower() != 'nan':
        name_parts.append(str(middle_name))
    name_parts.append(str(row['last_name']))
    full_name = ' '.join(name_parts)
    student_id = row['student_id']
    group = row.get('group', '')
    year = row.get('year', '')
    academic_year = row.get('academic year', '')
    email = row['email']
    if email is None or (isinstance(email, float) and np.isnan(email)):
        email = "No email provided"
    summary = f"{full_name} ({student_id}) is a student ({group}) in year {year} for the {academic_year} academic year. Their email is {email}."
    return summary

def search_by_name_fuzzy(name: str, threshold=60):
    corrected_name = correct_spelling(name)
    name_lower = corrected_name.strip().lower()
    student_names = student_df['first_name'].str.lower() + " " + student_df['last_name'].str.lower()
    student_matches = process.extract(name_lower, student_names, scorer=fuzz.token_sort_ratio, limit=10)
    matched_student_indices = [idx for _, score, idx in student_matches if score >= threshold]
    matched_students = student_df.iloc[matched_student_indices]

    instructor_df = load_instructor_data()
    instructor_names = instructor_df['instructor_name'].str.lower()
    instructor_matches = process.extract(name_lower, instructor_names, scorer=fuzz.token_sort_ratio, limit=10)
    matched_instructor_indices = [idx for _, score, idx in instructor_matches if score >= threshold]
    matched_instructors = instructor_df.iloc[matched_instructor_indices]

    return matched_students, matched_instructors

def parse_student_filters(query: str):
    filters = {}
    query_lower = query.lower()
    fields = ['student_id', 'first_name', 'middle_name', 'last_name', 'gender', 'group', 'semester', 'year', 'academic year', 'email']
    for field in fields:
        if field in query_lower:
            import re
            pattern = rf"{field}\s*=\s*([\w@.\-]+)"
            match = re.search(pattern, query_lower)
            if match:
                filters[field] = match.group(1)
    return filters

def filter_students(filters: dict):
    df = student_df.copy()
    for field, value in filters.items():
        df = df[df[field].str.lower() == value.lower()]
    return df

if st.button("Send"):
    if prompt.strip():
        # --- Special case: Who made/created/built this chatbot ---
        creator_questions = [
            "who built this chat bot", "who build this chat bot", "who built this chatbot", "who build this chatbot",
            "who create this chatbot", "who created this chatbot", "who create this chat bot", "who created this chat bot",
            "who make this chat", "who make this chatbot", "who made this chat", "who made this chatbot",
            "who create this chat", "who created this chat"
        ]
        if prompt.strip().lower() in creator_questions:
            response_text = "I was built by me THY CHANTHA."
            st.session_state.chat_history.append({"role": "bot", "content": response_text})
            st.empty().markdown(response_text)
            log_evaluation(prompt, response_text)
        else:
            try:
                if prompt.strip().lower() in ["list all students", "list all student", "show all students", "show all student"]:
                    if student_df.empty:
                        st.info("No student data available.")
                        log_evaluation(prompt, "No student data available.")
                    else:
                        display_df = student_df[['student_id', 'first_name', 'middle_name', 'last_name', 'email']]
                        display_df = display_df.fillna('')
                        markdown_table = display_df.to_markdown(index=False)
                        st.markdown(f"### Student List\n\n{markdown_table}")
                        st.session_state.chat_history.append({"role": "bot", "content": f"### Student List\n\n{markdown_table}"})
                        log_evaluation(prompt, "Displayed student list.")
                elif prompt.strip().lower() in ["list all instructors", "list all instructor", "show all instructors", "show all instructor"]:
                    try:
                        instructor_df = load_instructor_data()
                        if instructor_df.empty:
                            st.info("No instructor data available.")
                            log_evaluation(prompt, "No instructor data available.")
                        else:
                            display_df = instructor_df[['instructor_id', 'department_id', 'instructor_name', 'email']].fillna('')
                            markdown_table = display_df.to_markdown(index=False)
                            st.markdown(f"### Instructor List\n\n{markdown_table}")
                            st.session_state.chat_history.append({"role": "bot", "content": f"### Instructor List\n\n{markdown_table}"})
                            log_evaluation(prompt, "Displayed instructor list.")
                    except Exception as e:
                        st.error(f"Error loading instructor data: {e}")
                        log_evaluation(prompt, f"Error loading instructor data: {e}")
                else:
                    filters = parse_student_filters(prompt)
                    if filters:
                        filtered_students = filter_students(filters)
                        if not filtered_students.empty:
                            sentences = []
                            for _, row in filtered_students.iterrows():
                                name_parts = [str(row['first_name'])]
                                middle_name = row.get('middle_name', '')
                                if middle_name and str(middle_name).lower() != 'nan':
                                    name_parts.append(str(middle_name))
                                name_parts.append(str(row['last_name']))
                                full_name = ' '.join(name_parts)
                                email = row['email']
                                if email is None or (isinstance(email, float) and np.isnan(email)):
                                    email = "No email provided"
                                sentence = f"{full_name} (ID: {row['student_id']}) can be contacted at {email}."
                                sentences.append(sentence)
                            response_text = "\n\n".join(sentences)
                            st.markdown(f"### Students matching filters\n\n{response_text}")
                            st.session_state.chat_history.append({"role": "bot", "content": f"### Students matching filters\n\n{response_text}"})
                            log_evaluation(prompt, f"Displayed students matching filters with sentences.")
                        else:
                            # Removed the "No students found matching..." message as requested
                            # st.info("No students found matching the filters.")
                            # st.session_state.chat_history.append({"role": "bot", "content": "No students found matching the filters."})
                            log_evaluation(prompt, "No students found matching the filters.")
                    else:
                        if len(prompt.strip().split()) == 1:
                            name = prompt.strip()
                            matched_students, matched_instructors = search_by_name_fuzzy(name)
                            if not matched_students.empty:
                                sentences = []
                                if len(matched_students) == 1:
                                    row = matched_students.iloc[0]
                                    response_text = format_student_summary(row)
                                    st.markdown(f"### Student Summary\n\n{response_text}")
                                    st.session_state.chat_history.append({"role": "bot", "content": f"### Student Summary\n\n{response_text}"})
                                    log_evaluation(prompt, response_text)
                                else:
                                    for _, row in matched_students.iterrows():
                                        name_parts = [str(row['first_name'])]
                                        middle_name = row.get('middle_name', '')
                                        if middle_name and str(middle_name).lower() != 'nan':
                                            name_parts.append(str(middle_name))
                                        name_parts.append(str(row['last_name']))
                                        full_name = ' '.join(name_parts)
                                        email = row['email']
                                        if email is None or (isinstance(email, float) and np.isnan(email)):
                                            email = "No email provided"
                                        sentence = f"{full_name} (ID: {row['student_id']}) can be contacted at {email}."
                                        sentences.append(sentence)
                                    response_text = "\n\n".join(sentences)
                                    st.markdown(f"### Students matching name '{name}'\n\n{response_text}")
                                    st.session_state.chat_history.append({"role": "bot", "content": f"### Students matching name '{name}'\n\n{response_text}"})
                                    log_evaluation(prompt, f"Displayed students matching name '{name}' with sentences.")
                            elif not matched_instructors.empty:
                                sentences = []
                                for _, row in matched_instructors.iterrows():
                                    instructor_name = str(row.get('instructor_name', 'Unknown'))
                                    if not instructor_name or instructor_name.lower() == 'nan':
                                        instructor_name = 'Unknown'
                                    email = row['email']
                                    if email is None or (isinstance(email, float) and np.isnan(email)):
                                        email = "No email provided"
                                    sentence = f"{instructor_name} (ID: {row['instructor_id']}) from department {row['department_id']} can be contacted at {email}."
                                    sentences.append(sentence)
                                response_text = "\n\n".join(sentences)
                                st.markdown(f"### Instructors matching name '{name}'\n\n{response_text}")
                                st.session_state.chat_history.append({"role": "bot", "content": f"### Instructors matching name '{name}'\n\n{response_text}"})
                                log_evaluation(prompt, f"Displayed instructors matching name '{name}' with sentences.")
                            else:
                                # Removed the message below as requested
                                # st.info(f"No students or instructors found matching '{name}'.")
                                # st.session_state.chat_history.append({"role": "bot", "content": f"No students or instructors found matching '{name}'."})
                                log_evaluation(prompt, f"No matches for name '{name}'.")
                        else:
                            # Removed the message below as requested
                            # st.info("No students or instructors found matching the prompt.")
                            # st.session_state.chat_history.append({"role": "bot", "content": "No students or instructors found matching the prompt."})
                            log_evaluation(prompt, "No matches for prompt.")

                        relevant_texts = find_relevant_texts(prompt, vectorizer, tfidf_matrix, combined_texts)
                        if not relevant_texts:
                            context = "No relevant data found."
                        else:
                            context = "\n".join(relevant_texts)
                        augmented_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(augmented_prompt)
                        response_text = response.text
                        if response_text.strip():
                            st.markdown(response_text)
                            st.session_state.chat_history.append({"role": "bot", "content": response_text})
                            log_evaluation(prompt, response_text)
                        else:
                            st.info("No clear answer was generated for the given prompt.")
                            st.session_state.chat_history.append({"role": "bot", "content": "No clear answer was generated for the given prompt."})
                            log_evaluation(prompt, "No clear answer generated.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
                st.session_state.chat_history.append({"role": "bot", "content": f"❌ An unexpected error occurred: {e}"})
                log_evaluation(prompt, f"Error: {e}")

# Show evaluation log for manual review
if st.checkbox("Show Evaluation Log"):
    st.write(st.session_state.eval_log)

st.markdown("""
### Feedback Loop
- Collect user feedback on answers.
- Use feedback to improve data quality, embeddings, and prompt design iteratively.
""")
