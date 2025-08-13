# AI Resume Optimiser & Generator

A **Streamlit app** that optimises resumes to be ATS-friendly and tailored to specific job descriptions. It analyses your resume, extracts keywords, and produces a **refined DOCX resume** along with a **change log** highlighting improvements.

---

## Features

* Upload resume in **PDF or DOCX** format
* Input job description via **text paste or file upload**
* Extracts and aligns **keywords** with the job description
* Generates **ATS-optimised resume** in DOCX format
* Produces a **concise change log** (keywords, formatting, ATS checks)

---

## Installation & Running

1. Clone the repository
2. Create and activate a virtual environment (Python 3.10+)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run main.py
```

5. Open the local URL in your browser (default: `http://localhost:8501`)

---

## Notes

* Focused on **ATS-friendly text**, avoiding tables/images.
* Keeps consistent section headers: `Summary`, `Skills`, `Experience`, `Projects`, `Education`.
* Injects missing **JD keywords** into Skills.
* Reorders Experience bullets by **relevance to JD**.

---

## Dependencies

* `streamlit`
* `pdfplumber`
* `python-docx`
* `scikit-learn`
* `nltk`
* `rapidfuzz`

Optional: `spacy` for advanced NLP

If you want, I can also **write a 3–4 line project description** suitable for your GitHub portfolio homepage. Do you want me to do that?
