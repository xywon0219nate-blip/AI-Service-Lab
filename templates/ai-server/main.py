from fastapi import FastAPI

app = FastAPI(
    title="AI Service Lab API",
    description="Master FastAPI template for AI services",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "AI Service Blueprint API is running",
        "status": "success"
    }

@app.get("/health")
def health():
    return {
        "status": "OK"
    }