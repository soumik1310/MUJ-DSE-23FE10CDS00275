from pathlib import Path

import joblib

from preprocess import clean_text

MODELS_DIR = Path(__file__).parent / "models"
cache = {}


def load():
    if "vec" not in cache:
        cache["vec"] = joblib.load(MODELS_DIR / "vectorizer.joblib")
        cache["model"] = joblib.load(MODELS_DIR / "model.joblib")
    return cache["vec"], cache["model"]


def predict_style(text):
    try:
        vec, model = load()
    except FileNotFoundError:
        return {"available": False, "message": "Model files not found. Run fake_news_model.ipynb first."}

    cleaned = clean_text(text)
    if not cleaned:
        return {"available": False, "message": "Not enough usable text for the ML model."}

    proba = model.predict_proba(vec.transform([cleaned]))[0]
    fake_prob = float(proba[list(model.classes_).index(1)])
    return {
        "available": True,
        "label": "fake-style" if fake_prob >= 0.5 else "real-style",
        "confidence": round(max(fake_prob, 1 - fake_prob), 3),
        "fake_probability": round(fake_prob, 3),
    }


def explain_style(text, top_k=8, max_words=60):
    try:
        vec, _ = load()
        if "explainer" not in cache:
            cache["explainer"] = joblib.load(MODELS_DIR / "explainer.joblib")
            cache["names"] = vec.get_feature_names_out()
    except FileNotFoundError:
        return None

    cleaned = clean_text(text)
    X = vec.transform([cleaned]).tocsr() if cleaned else None
    if X is None or X.nnz == 0:
        return None

    contrib = cache["explainer"].coef_[0][X.indices] * X.data
    items = sorted(zip(cache["names"][X.indices], contrib), key=lambda t: t[1])

    def fmt(pairs):
        return [{"word": str(w), "weight": round(float(c), 4)} for w, c in pairs]

    top_real = fmt([t for t in items if t[1] < 0][:top_k])
    top_fake = fmt([t for t in reversed(items) if t[1] > 0][:top_k])
    singles = sorted((t for t in items if " " not in t[0]), key=lambda t: -abs(t[1]))[:max_words]
    return {"top_fake": top_fake, "top_real": top_real,
            "word_weights": {str(w): round(float(c), 4) for w, c in singles}}
