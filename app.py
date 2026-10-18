import os
import time
import streamlit as st
from google import genai
from google.genai import types
import pypdf
import docx

# Page Configuration
st.set_page_config(
    page_title="ATS Resume Analyzer & Evaluator",
    page_icon="📄",
    layout="wide"
)

# Function to extract text from PDF files
def extract_text_from_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Extract text based on file format
def get_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file format.")

# Call Gemini Flash Model with Automatic Retry for 503 Errors
def evaluate_resume(resume_text, job_description="", api_key=""):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert HR Specialist and ATS (Applicant Tracking System) Evaluation Engine.
    Analyze the provided resume and return a structured report with clear sections.

    Resume Text:
    {resume_text}

    Job Description (If provided, score match against this; if blank, evaluate overall ATS compatibility):
    {job_description if job_description else "General Industry Standard Resume Analysis"}

    Please format your response clearly in Markdown with the following structured sections:
    
    1. **Estimated ATS Score**:
       - Provide an overall ATS score out of 100 with a brief summary of how the score was calculated.

    2. **Key Strengths**:
       - Highlight strong sections, skills, bullet points, or quantifiable achievements.

    3. **Critical Areas for Improvement**:
       - Detail formatting issues, missing keywords, weak phrasing, or passive voice.

    4. **Actionable Suggestions & Rewrites**:
       - Give concrete examples of how to rewrite specific bullet points to improve impact.

    5. **Missing Keywords / Skills**:
       - List key hard and soft skills that should be added.
    """
    
    max_retries = 3
    backoff_factor = 2
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            # If it's a 503 / temporary overload error and we haven't maxed out retries
            if ("503" in error_str or "UNAVAILABLE" in error_str) and attempt < max_retries - 1:
                sleep_time = backoff_factor ** (attempt + 1) # 2s, 4s...
                time.sleep(sleep_time)
                continue
            else:
                raise e

# --- UI Header ---
st.title("📄 ATS Resume Analyzer & Score Booster")
st.write("Upload your resume and get instant feedback, ATS scoring, and targeted suggestions powered by Gemini Flash.")

# Sidebar for API key configuration
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Get an API key from Google AI Studio"
)

# Retrieve key from Streamlit secrets if available, fallback to sidebar input
gemini_api_key = st.secrets.get("GEMINI_API_KEY", api_key_input)

if not gemini_api_key:
    st.sidebar.warning("⚠️ Please enter your Gemini API Key or configure it in secrets.")

# Inputs
uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_desc = st.text_area("Target Job Description (Optional)", placeholder="Paste the job description here for matched evaluation...", height=150)

# Main Processing Button
if st.button("Evaluate Resume", type="primary"):
    if not gemini_api_key:
        st.error("Missing API Key! Enter your Gemini API Key in the sidebar or setup Streamlit Secrets.")
    elif not uploaded_file:
        st.warning("Please upload a resume file first.")
    else:
        with st.spinner("Extracting text and analyzing resume with Gemini Flash (retrying automatically if busy)..."):
            try:
                resume_text = get_resume_text(uploaded_file)
                
                if not resume_text.strip():
                    st.error("Unable to extract text from the uploaded file. Please ensure it is not an image-only PDF.")
                else:
                    feedback = evaluate_resume(resume_text, job_desc, gemini_api_key)
                    st.success("Analysis Complete!")
                    st.markdown(feedback)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
