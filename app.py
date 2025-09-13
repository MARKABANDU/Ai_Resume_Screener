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

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# ----------- PDF TEXT EXTRACTION -----------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ----------- PREPROCESS TEXT -----------
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# ----------- ENTITY EXTRACTION -----------
def extract_entities(text):
    doc = nlp(text)
    email = [token.text for token in doc if token.like_email]
    phone = [token.text for token in doc if token.like_num and len(token.text) >= 10]
    name = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return name[0] if name else "N/A", email[0] if email else "N/A", phone[0] if phone else "N/A"

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
    missing = list(job_tokens - resume_tokens)
    return ", ".join(missing[:10])

# ----------- WORDCLOUD VISUALIZATION -----------
def show_wordcloud(text):
    if text.strip() == "":
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ----------- STREAMLIT APP -----------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screener")
st.write("Upload resumes and match them against a Job Description.")

# Job Description Input
job_desc = st.text_area("Paste Job Description here")

# Resume Upload
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

# Matching Threshold
threshold = st.slider("Set Matching Threshold (%)", 0, 100, 70)

if uploaded_files and job_desc:
    resumes_text = []
    names = []
    details = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        clean_text = preprocess_text(text)
        resumes_text.append(clean_text)
        names.append(file.name)

        # Extract details
        candidate_name, email, phone = extract_entities(text)
        details.append((candidate_name, email, phone))

    # Rank resumes
    scores, tfidf = rank_resumes(preprocess_text(job_desc), resumes_text)

    results = []
    for i, resume_text in enumerate(resumes_text):
        missing_keywords = find_missing_keywords(preprocess_text(job_desc), resume_text, tfidf)
        results.append({
            "Resume File": names[i],
            "Candidate Name": details[i][0],
            "Email": details[i][1],
            "Phone": details[i][2],
            "Score (%)": round(scores[i]*100, 2),
            "Recommendation": "Needs Improvement" if scores[i]*100 < threshold else "Good Match",
            "Missing Keywords": missing_keywords
        })

    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)

    st.subheader("📊 Resume Match Results")
    st.dataframe(df, use_container_width=True)

    # ----------- Visualization -----------
    st.subheader("📈 Score Distribution")
    fig = px.bar(df, x="Resume File", y="Score (%)", color="Recommendation", title="Resume Match Scores")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("☁ Missing Keywords Wordcloud")
    all_missing = " ".join(df["Missing Keywords"].tolist())
    show_wordcloud(all_missing)

    # ----------- Download option -----------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Results as CSV", data=csv, file_name="resume_ranking.csv", mime="text/csv")

    # ----------- Save to Database -----------
    conn = sqlite3.connect("resume_results.db")
    df.to_sql("results", conn, if_exists="append", index=False)
    conn.close()
    st.success("Results saved to database ✅")
