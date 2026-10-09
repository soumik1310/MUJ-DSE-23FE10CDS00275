# Installation

You need Python 3.10 or newer and a free Gemini API key from https://aistudio.google.com/apikey

Datasets:
- ISOT (`Fake.csv`, `True.csv`): https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- LIAR `train.tsv` (optional, for the generalization test)

## Steps

```bash
git clone <repo-url>
cd <repo>/capstone
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac or Linux use `source venv/bin/activate` instead of `venv\Scripts\activate`.

1. Put `Fake.csv` and `True.csv` in the `data/` folder.
2. Open `fake_news_model.ipynb` and run all cells. It creates the `models/` files and the charts in `assets/`.
3. Copy `.env.example` to `.env` and put your key after `GEMINI_API_KEY=`.
4. Run `python app.py` and open http://127.0.0.1:8000

## If something fails

- "Model files not found": run the notebook first.
- "LLM call failed": check the key, or change `GEMINI_MODEL` in `.env`.
- Never commit `.env`. It is already in `.gitignore`.
