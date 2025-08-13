import re
from typing import List, Dict
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Regex for contact info
CONTACT_PATTERN = re.compile(
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})|"
    r"(?P<phone>\+?\d[\d\s\-()]{7,}\d)|"
    r"(?P<linkedin>linkedin\.com/in/[A-Za-z0-9\-_/]+)|"
    r"(?P<github>github\.com/[A-Za-z0-9\-_/]+)",
    re.I
)

# ------------------ Keyword Extraction ------------------ #
def tfidf_keywords(text: str, top_k: int = 50) -> List[str]:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=2000, min_df=1)
    X = vectorizer.fit_transform([text])
    terms = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    scored = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    # Filter short/number-only tokens
    return [t for t, s in scored if len(t) > 2 and not t.isdigit()][:top_k]

# ------------------ Keyword Match Accuracy ------------------ #
def keyword_match(jd_text: str, resume_text: str) -> float:
    jd_kws = set(tfidf_keywords(jd_text, top_k=100))
    resume_kws = set(tfidf_keywords(resume_text, top_k=150))

    matched = 0
    for kw in jd_kws:
        for r_kw in resume_kws:
            if fuzz.partial_ratio(kw, r_kw) >= 90:
                matched += 1
                break
    if len(jd_kws) == 0:
        return 0.0
    return (matched / len(jd_kws)) * 100

# ------------------ ATS Compatibility Score ------------------ #
def ats_score(resume_text: str) -> float:
    score = 0
    total = 4  # number of checks

    # Contact info
    if CONTACT_PATTERN.search(resume_text):
        score += 1

    # Word count check
    words = len(resume_text.split())
    if 250 <= words <= 1200:
        score += 1

    # Avoid tables
    if '|' not in resume_text:
        score += 1

    # Avoid fancy symbols
    if not re.search(r"[■◆●◦★]", resume_text):
        score += 1

    return (score / total) * 100

# ------------------ Example Usage ------------------ #
if __name__ == "__main__":
    # Load your files
    with open("job_description.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()

    with open("optimised_resume.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    kw_accuracy = keyword_match(jd_text, resume_text)
    ats_compat = ats_score(resume_text)

    print(f"✅ Keyword Match Accuracy: {kw_accuracy:.2f}%")
    print(f"✅ ATS Compatibility Score: {ats_compat:.2f}%")
