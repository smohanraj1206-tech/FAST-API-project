import time
from fastapi import APIRouter, HTTPException, status
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from model.predict import predictor_instance

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Returns API health status and ML model load state.
    """
    return HealthResponse(
        status="healthy",
        message="FastAPI Backend & ML Model are operating normally.",
        model_loaded=(predictor_instance.model is not None),
        version="1.0.0"
    )

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_performance(request: PredictionRequest):
    """
    Main prediction endpoint. Accepts student study habits and returns
    predicted GPA, performance level, model confidence, weak area analysis,
    personalized AI recommendations, and weekly study plan.
    """
    try:
        time.sleep(0.15)
        return predictor_instance.predict(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction calculation: {str(e)}"
        )

@router.post("/analyze-habits", response_model=PredictionResponse, tags=["Analysis"])
def analyze_habits(request: PredictionRequest):
    """
    Alias endpoint for habit analysis and performance prediction.
    """
    return predict_performance(request)
