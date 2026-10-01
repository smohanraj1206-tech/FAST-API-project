from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="AI-Powered Study Habit Analyzer & Performance Predictor API",
    description="Machine Learning REST API for academic performance prediction, weak area detection, and AI recommendations.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(router)

@app.get("/", tags=["General"])
def read_root():
    return {
        "service": "AI-Powered Study Habit Analyzer & Performance Predictor API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "analyze_habits": "/analyze-habits"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
