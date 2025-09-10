import streamlit as st
st.title("Hello Bro 👋 App Work Aagudhu!")
import streamlit as st
import pandas as pd
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# ----------- PDF TEXT EXTRACTION -----------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ----------- PREPROCESS TEXT -----------
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# ----------- RANK RESUMES -----------
def rank_resumes(job_desc, resumes):
    documents = [job_desc] + resumes
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return scores, tfidf

# ----------- FIND MISSING KEYWORDS -----------
def find_missing_keywords(job_desc, resume, tfidf):
    job_tokens = set(job_desc.split())
    resume_tokens = set(resume.split())
    missing = job_tokens - resume_tokens
    return ", ".join(list(missing)[:10])  # limit top 10 keywords
# ----------- STREAMLIT APP -----------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screener")
st.write("Upload resumes and match them against a Job Description.")

# Job Description Input
job_desc = st.text_area("Paste Job Description here")

# Resume Upload
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files and job_desc:
    resumes_text = []
    names = []
    emails = []  # optional: detect email from resume if needed later
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        clean_text = preprocess_text(text)
        resumes_text.append(clean_text)
        names.append(file.name)
    
    # Rank resumes
    scores, tfidf = rank_resumes(preprocess_text(job_desc), resumes_text)
    
    results = []
    for i, resume_text in enumerate(resumes_text):
        missing_keywords = find_missing_keywords(preprocess_text(job_desc), resume_text, tfidf)
        results.append({
            "Resume File": names[i],
            "Score (%)": round(scores[i]*100, 2),
            "Recommendation": "Needs Improvement" if scores[i] < 60 else "Good Match",
            "Missing Keywords": missing_keywords
        })
    
    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)
    
    st.subheader("📊 Resume Match Results")
    st.dataframe(df, use_container_width=True)
    
    # Download option
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Results as CSV", data=csv, file_name="resume_ranking.csv", mime="text/csv")
   