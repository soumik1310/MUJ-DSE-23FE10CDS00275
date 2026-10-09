from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llm_verify import verify_with_gemini
from ml_predict import explain_style, predict_style

BASE_DIR = Path(__file__).parent
app = FastAPI(title="Fake News Detector")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=422, detail="Text is too short.")
    ml = predict_style(text)
    if ml.get("available"):
        ml["explanation"] = explain_style(text)
    return {"ml": ml, "llm": verify_with_gemini(text)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
