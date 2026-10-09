import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

DATELINE = r"(?:\b[A-Z][A-Z .,/'-]{1,40})?\(\s*[Rr]euters\s*\)\s*-?\s*"
LEAKS = r"\breuters\b|21st century wire|featured image|getty images|image via"
LINKS = r"http\S+|www\.\S+|pic\.twitter\.com\S*"


def clean_text(text):
    text = re.sub(DATELINE, " ", str(text))
    text = re.sub(LINKS, " ", text).lower()
    text = re.sub(LEAKS, " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

