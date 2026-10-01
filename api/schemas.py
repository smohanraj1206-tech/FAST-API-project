from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    # Core study habit metrics
    study_hours: float = Field(..., ge=0, le=16, description="Average study hours per day")
    sleep_hours: float = Field(..., ge=0, le=16, description="Average sleep hours per night")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100)")
    assignment_completion: float = Field(..., ge=0, le=100, description="Assignment completion percentage (0-100)")
    previous_gpa: float = Field(..., ge=0, le=10.0, description="Previous semester GPA (0.0 - 10.0)")
    screen_time: float = Field(..., ge=0, le=18, description="Average daily screen time in hours")
    self_study_hours: float = Field(..., ge=0, le=16, description="Self study hours per day")
    revision_frequency: float = Field(..., ge=0, le=7, description="Revision frequency in days per week")
    test_frequency: float = Field(..., ge=0, le=7, description="Practice or test frequency in days per week")

    # Optional student metadata
    student_name: Optional[str] = Field("Student", description="Student Name")
    age: Optional[int] = Field(20, ge=15, le=60, description="Age")
    academic_year: Optional[str] = Field("Junior", description="Academic Year")
    department: Optional[str] = Field("General Science", description="Department/Major")
    study_days: Optional[float] = Field(5, ge=1, le=7, description="Study days per week")
    num_subjects: Optional[int] = Field(5, ge=1, le=12, description="Number of subjects")
    exercise_hours: Optional[float] = Field(3, ge=0, le=24, description="Exercise hours per week")
    class_participation: Optional[float] = Field(7, ge=1, le=10, description="Class participation score (1-10)")


class WeakAreaItem(BaseModel):
    area: str
    problem: str
    current_value: str
    target_value: str
    severity: str  # "high", "medium", "low"
    icon: str


class RecommendationItem(BaseModel):
    category: str  # "Study Plan", "Sleep", "Revision", "Screen Time", "Practice"
    recommendation: str
    priority: str  # "High", "Medium", "Low"
    icon: str


class DaySchedule(BaseModel):
    day: str
    study_hours: float
    subject: str
    revision: str
    test_practice: str


class PredictionResponse(BaseModel):
    predicted_gpa: float
    performance_level: str  # "Poor", "Average", "Good", "Excellent"
    confidence: float       # Percentage 0-100
    weak_areas: List[WeakAreaItem]
    recommendations: List[RecommendationItem]
    weekly_plan: Optional[List[DaySchedule]] = None
    habit_radar_data: Optional[List[Dict[str, Any]]] = None
    study_vs_performance_chart: Optional[List[Dict[str, Any]]] = None
    weekly_pattern_data: Optional[List[Dict[str, Any]]] = None
    input_summary: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    version: str
