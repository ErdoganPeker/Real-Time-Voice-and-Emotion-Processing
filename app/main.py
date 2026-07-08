from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import os

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DEMO = [
    {"text": "Hello, how are you today?", "emotion": "Happy", "confidence": 0.87, "emotions": {"Happy": 87, "Neutral": 8, "Surprised": 5}},
    {"text": "I am a bit tired from all the work.", "emotion": "Sad", "confidence": 0.73, "emotions": {"Sad": 73, "Neutral": 20, "Tired": 7}},
    {"text": "This AI project is incredible!", "emotion": "Excited", "confidence": 0.92, "emotions": {"Excited": 92, "Happy": 6, "Neutral": 2}},
    {"text": "Let me check the model accuracy now.", "emotion": "Neutral", "confidence": 0.81, "emotions": {"Neutral": 81, "Focused": 12, "Calm": 7}},
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/demo/{step}")
async def demo_step(step: int):
    return DEMO[step % len(DEMO)]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5015)
