from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers import clarity, inference, test_models, training

app = FastAPI(title="LLM Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router, prefix="/api/training")
app.include_router(inference.router, prefix="/api/inference")
app.include_router(test_models.router, prefix="/api/test")
app.include_router(clarity.router, prefix="/api/clarity")


@app.get("/health")
def health():
    return {"status": "ok"}
