# Installation

You need Python 3.10 or newer and a free Gemini API key from https://aistudio.google.com/apikey

Datasets:
- ISOT (`Fake.csv`, `True.csv`): https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- LIAR `train.tsv` (optional, for the generalization test)

## Steps

```bash
git clone https://github.com/soumik1310/MUJ-DSE-23FE10CDS00275.git
cd MUJ-DSE-23FE10CDS00275/code
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac or Linux use `source venv/bin/activate` instead of `venv\Scripts\activate`.

1. Put `Fake.csv` and `True.csv` in `code/data/`.
2. Go to the `notebooks/` folder, open `fake_news_model.ipynb` and run all cells. It creates the model files in `code/models/` and the charts in `capstone/assets/`.
3. In `code/`, copy `.env.example` to `.env` and put your key after `GEMINI_API_KEY=`.
4. From `code/`, run `python app.py` and open http://127.0.0.1:8000

## If something fails

- "Model files not found": run the notebook first.
- "LLM call failed": check the key, or change `GEMINI_MODEL` in `.env`.
- Never commit `.env`. It is already in `.gitignore`.
