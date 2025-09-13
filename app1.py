import streamlit as st
import pandas as pd
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sqlite3

@st.cache_resource
def load_nlp_model(model_name="en_core_web_sm"):
    """Loads a spaCy model, providing instructions if it's not found."""
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"SpaCy model '{model_name}' not found.")
        st.info(f"Please download the model by running this command in your terminal:\n\npython -m spacy download {model_name}")
        st.stop()

# Load NLP model
nlp = load_nlp_model()

# ----------- PDF TEXT EXTRACTION -----------
def extract_text_from_pdf(file):
    text = ""
    num_pages = 0
    with pdfplumber.open(file) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text, num_pages

# ----------- PREPROCESS TEXT -----------
def preprocess_text(text):
    """Cleans and lemmatizes text, preserving technical terms."""
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        # Keep tokens that are alphabetic, or are alpha-numeric with special chars common in tech (e.g., C++, .NET)
        if not token.is_stop and not token.is_punct and (token.is_alpha or token.is_ascii and any(char.isdigit() or char in ['+', '#', '.'] for char in token.text)):
            tokens.append(token.lemma_)
    return " ".join(tokens)

# ----------- ENTITY EXTRACTION -----------
def extract_entities(text):
    """Extracts name, email, and phone with improved accuracy for names."""
    full_doc = nlp(text)
    # Prioritize names from the beginning of the document
    name_doc = nlp(text[:300]) 
    name = [ent.text for ent in name_doc.ents if ent.label_ == "PERSON"]
    email = [token.text for token in full_doc if token.like_email]
    phone = [token.text for token in full_doc if token.like_num and len(token.text) >= 10]
    return name[0] if name else "N/A", email[0] if email else "N/A", phone[0] if phone else "N/A"

# ----------- RANK RESUMES -----------
def rank_resumes(job_desc, resumes):
    documents = [job_desc] + resumes
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return scores, tfidf

# ----------- FIND MISSING KEYWORDS -----------
def find_missing_keywords(job_desc_processed, resume_processed, tfidf_vectorizer):
    """Finds important missing keywords from the resume compared to the job description."""
    # Create a dictionary mapping feature names (vocabulary) to their index
    vocab = tfidf_vectorizer.vocabulary_

    # Get the tf-idf vector for the job description
    job_desc_vec = tfidf_vectorizer.transform([job_desc_processed])

    job_tokens = set(job_desc_processed.split())
    resume_tokens = set(resume_processed.split())
    missing_tokens_set = job_tokens - resume_tokens

    # Rank missing tokens by their TF-IDF score in the job description
    missing_keywords_with_scores = {}
    for token in missing_tokens_set:
        if token in vocab:
            token_index = vocab[token]
            score = job_desc_vec[0, token_index]
            if score > 0:
                missing_keywords_with_scores[token] = score
    
    sorted_missing_keywords = sorted(missing_keywords_with_scores.keys(), key=lambda x: missing_keywords_with_scores[x], reverse=True)
    # Capitalize keywords for better readability
    proper_keywords = [keyword.capitalize() for keyword in sorted_missing_keywords[:10]]
    return ", ".join(proper_keywords)

# ----------- WORDCLOUD VISUALIZATION -----------
def show_wordcloud(text):
    if text.strip() == "":
        return
    # Use a dark background for the word cloud to match the theme
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ----------- STYLING -----------
def style_recommendation(val):
    """Styles the 'Recommendation' column based on its value."""
    if val == 'Needs Improvement':
        return 'background-color: #9C0006; color: white'  # Dark red background with white text
    elif val == 'Good Match':
        return 'background-color: #006100; color: white'  # Dark green background with white text
    return ''

# ----------- STREAMLIT APP -----------
st.set_page_config(page_title="AI Resume Screener", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Professional Theme ---
st.markdown("""
<style>
    /* Main background color */
    .stApp {
        background-color: #D6EAF8; /* A more distinct Light Blue Background */
    }
    /* Text color for the entire app */
    .stApp, .stApp p, .stApp li, .stApp label {
        color: #1A253C; /* Dark Blue Text */
    }
    h1, h2, h3 {
        color: #1A253C; /* Dark Blue for headers */
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #EBF5FB; /* Lighter blue for sidebar */
    }
    /* Main title styling */
    h1 {
        text-align: center;
        background-color: #2E86C1; /* Professional blue background */
        color: white !important; /* White text for contrast */
        padding: 0.75rem;
        border-radius: 10px;
    }
    /* Tab styling to make the active tab stand out */
    [data-testid="stTabs"] button {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #FFFFFF;
        color: #2E86C1;
        font-weight: bold;
    }
    /* Text Area ("Box") styling */
    [data-testid="stTextArea"] textarea {
        background-color: #FFFFFF;
        color: #1A253C; /* Ensure text inside the box is dark blue */
        border: 1px solid #B0C4DE;
        caret-color: #1A253C; /* Make blinking cursor visible */
    }
    /* Style for uploaded file names and size text */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        color: #000000; /* Changed to black for better visibility */
    }
    /* Sidebar collapse button color */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        color: #2E86C1;
    }
    /* Sidebar Section Titles */
    [data-testid="stSidebar"] h1 {
        color: #2E86C1; /* Use a distinct blue for sidebar titles */
    }
    /* Button styling for "Process" and "Browse files" */
    div[data-testid="stButton"] > button:not([kind="primary"]), [data-testid="stFileUploader"] section button, [data-testid="stDownloadButton"] button {
        background-color: #2E86C1; /* Professional blue */
        color: #FFFFFF; /* White text */
    }
    div[data-testid="stButton"] > button:not([kind="primary"]):hover, [data-testid="stFileUploader"] section button:hover, [data-testid="stDownloadButton"] button:hover {
        background-color: #21618C; /* Darker blue on hover */
        color: #FFFFFF; /* Change to pure white on hover for a nice effect */
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    threshold = st.slider("Set Matching Threshold (%)", 0, 100, 70, help="Set the minimum score for a resume to be considered a 'Good Match'.")
    
    st.title("ℹ️ Overview")
    st.markdown("""
    This AI-powered Resume Screener helps recruiters and HR professionals quickly identify the best-matching candidates for a given job description.

    **✅ Key Features:**
    *   Automated Screening – Matches resumes against job descriptions using NLP.
    *   Keyword Relevance – Uses TF-IDF to identify important skills and terms.
    *   Candidate Information Extraction – Leverages spaCy for parsing candidate details.
    *   Customizable Threshold – Adjust matching percentage to refine results.
    *   Visual Insights – Provides score-based ranking and easy-to-understand visualizations.
    *   Bulk Resume Support – Upload multiple candidate resumes in PDF format.

    **📊 Use Case Example:**
    *   Recruiter pastes job description.
    *   Uploads candidate resumes.
    *   App ranks resumes by similarity score.
    *   Recruiter downloads results or reviews top candidates.
    """)

# --- Main App ---
st.markdown("<h1>📄 AI Resume Screener</h1>", unsafe_allow_html=True)

st.markdown("### 1. Paste Job Description")
job_desc = st.text_area("Paste Job Description here", height=250, label_visibility="collapsed")

st.markdown("### 2. Upload Candidate Resumes")
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

if st.button("✨ Analyze Resumes", use_container_width=True):
    if uploaded_files and job_desc:
        with st.spinner("Analyzing resumes... This might take a moment."):
            resumes_text = []
            names = []
            details = []
            pages_counts = []
            
            for file in uploaded_files:
                text, num_pages = extract_text_from_pdf(file)
                clean_text = preprocess_text(text)
                resumes_text.append(clean_text)
                names.append(file.name)
                
                # Extract details
                candidate_name, email, phone = extract_entities(text)
                details.append((candidate_name, email, phone))
                pages_counts.append(num_pages)
            
            # Preprocess job description once
            processed_job_desc = preprocess_text(job_desc)
            # Rank resumes
            scores, tfidf = rank_resumes(processed_job_desc, resumes_text)
            
            results = []
            for i, resume_text in enumerate(resumes_text):
                missing_keywords = find_missing_keywords(processed_job_desc, resume_text, tfidf)
                results.append({
                "Resume File": names[i],
                "Pages": pages_counts[i],
                "Candidate Name": details[i][0],
                "Email": details[i][1],
                "Phone": details[i][2],
                "Score (%)": round(scores[i]*100),
                "Recommendation": "Needs Improvement" if scores[i]*100 < threshold else "Good Match",
                "Missing Keywords": missing_keywords
                })
            
            df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)

        st.success("Analysis complete! See the results below.")

        # --- Output Tabs ---
        tab1, tab2, tab3 = st.tabs(["📊 Results Summary", "📈 Visualizations", "💾 Data Export"])

        with tab1:
            st.subheader("Resume Match Results")
            st.dataframe(df.style.applymap(style_recommendation, subset=['Recommendation']), use_container_width=True)

        with tab2:
            st.subheader("Score Distribution")
            # Use lighter colors for the dark theme chart
            color_map = {'Good Match': '#006100', 'Needs Improvement': '#9C0006'}
            fig = px.bar(df, x="Resume File", y="Score (%)", color="Recommendation", title="Resume Match Scores", color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Most Common Missing Keywords")
            all_missing = " ".join(df[df["Missing Keywords"].notna()]["Missing Keywords"].tolist())
            show_wordcloud(all_missing)

        with tab3:
            st.subheader("Download Results")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download as CSV", data=csv, file_name="resume_ranking.csv", mime="text/csv", use_container_width=True)
            
            st.subheader("Save to Database")
            if st.button("Save Results to DB", use_container_width=True, type="primary"):
                try:
                    with sqlite3.connect("resume_results.db") as conn:
                        df.to_sql("results", conn, if_exists="append", index=False)
                    st.success("Results successfully saved to database! ✅")
                except sqlite3.Error as e:
                    st.error(f"Database error: {e}")
        
        # Reset the processing state after completion
        st.session_state.processing = False
