from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import TicketRequest, TicketResponse
from predict import is_spam, predict_department
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os


ROOT_PATH = os.getenv("ROOT_PATH", "")
app = FastAPI(title="Lead Routing Customer Support", root_path=ROOT_PATH)

# allowing frontend to talk to api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # can restrict this to only actual frontend domain, like: allow_origins=["https://your-frontend-domain.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# serving frontend static files — html, css, js
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")

# serving index.html at root
@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))

# health check
@app.get("/health")
def health():
    return {"status": "ok"}

# prediction endpoint
@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest):
    text = request.user_input.strip()

    #checking spam first
    if is_spam(text):
        return TicketResponse(is_spam=True, department=None, confidence=None)

    #routing to department if ham
    department, confidence = predict_department(text)

    # low confidence means input is likely not a support query
    if department is None:
        return TicketResponse(is_spam=False, department="unrecognized", confidence=confidence)
    return TicketResponse(is_spam=False, department=department, confidence=confidence)