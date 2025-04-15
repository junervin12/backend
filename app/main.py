from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.analyzer import analyze_text

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze(request: TextRequest):
    result = analyze_text(request.text)
    return result
