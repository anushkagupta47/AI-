# AI Resume Optimiser & Generator (ATS-Friendly)

🧠 **AI-powered tool to optimise resumes for Applicant Tracking Systems (ATS)**

This Streamlit app helps candidates improve their resumes for specific job descriptions by:

* Parsing PDF or DOCX resumes
* Extracting and aligning keywords from resumes and job descriptions
* Generating an ATS-friendly, optimised resume (DOCX)
* Producing a concise change log highlighting improvements (keywords, formatting, ATS checks)

---

## Features

* ✅ Accepts **PDF and DOCX** resumes
* ✅ Keyword alignment with target Job Description (JD)
* ✅ Reorders experience and skills for maximum relevance
* ✅ Injects missing high-value skills into resume
* ✅ Produces **change log** showing ATS improvements
* ✅ Downloadable optimised resume

---

## Installation & Setup

### Requirements

* Python 3.10+
* Packages (install via pip):

```bash
pip install streamlit pdfplumber python-docx scikit-learn nltk rapidfuzz
```

Optional (for advanced NLP):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Run Locally

```bash
# Clone the repository
git clone <repository_url>
cd <repository_folder>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py
```

---

## Usage

1. Upload your **resume** (PDF or DOCX).
2. Paste or upload the **job description**.
3. Click **“Optimise Resume”**.
4. Download your **ATS-ready resume** and view the **change log**.

---

## Project Structure

```
AI-Resume-Optimiser/
│
├─ main.py              # Streamlit app
├─ accuracy_check.py    # Optional script to evaluate keyword alignment accuracy
├─ requirements.txt     # Python dependencies
├─ README.md            # Project documentation
└─ utils/               # Helper functions (parsing, keyword extraction, etc.)
```

---

## Notes

* Focuses on **ATS-friendly text**: avoid tables/images; simple layout.
* Supports **PDF and DOCX** formats (convert `.doc` to `.docx`).
* Generates summaries and skills sections tailored to the job description.

---

## License

MIT License ©
