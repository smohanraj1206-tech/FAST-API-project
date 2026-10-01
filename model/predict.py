import os
import pickle
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from api.schemas import PredictionRequest, PredictionResponse, WeakAreaItem, RecommendationItem, DaySchedule

MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

class AcademicPerformancePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = [
            "study_hours", "sleep_hours", "attendance", "assignment_completion",
            "previous_gpa", "screen_time", "self_study_hours",
            "revision_frequency", "test_frequency", "exercise_hours", "class_participation"
        ]
        self._initialize_or_load_model()

    def _initialize_or_load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    self.model = pickle.load(f)
                return
            except Exception:
                pass

        # If model.pkl is missing, trigger train.py
        from model.train import train_model
        self.model = train_model()

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        input_data = pd.DataFrame([{
            "study_hours": req.study_hours,
            "sleep_hours": req.sleep_hours,
            "attendance": req.attendance,
            "assignment_completion": req.assignment_completion,
            "previous_gpa": req.previous_gpa,
            "screen_time": req.screen_time,
            "self_study_hours": req.self_study_hours,
            "revision_frequency": req.revision_frequency,
            "test_frequency": req.test_frequency,
            "exercise_hours": req.exercise_hours or 3.0,
            "class_participation": req.class_participation or 7.0
        }])[self.feature_names]

        # ML Inference
        raw_prediction = float(self.model.predict(input_data)[0])
        predicted_gpa = round(float(np.clip(raw_prediction, 0.0, 10.0)), 2)

        # Performance Band
        if predicted_gpa >= 8.5:
            performance_level = "Excellent"
        elif predicted_gpa >= 7.2:
            performance_level = "Good"
        elif predicted_gpa >= 6.0:
            performance_level = "Average"
        else:
            performance_level = "Needs Improvement"

        # Model Confidence
        consistency_score = min(100.0, max(60.0, (
            (req.attendance * 0.3) + 
            (req.assignment_completion * 0.3) + 
            (min(1.0, req.study_hours / 6.0) * 40.0)
        )))
        confidence = round(80.0 + (consistency_score - 60.0) * 0.35, 1)
        confidence = min(96.0, max(75.0, confidence))

        # Detect Weak Areas
        weak_areas = self._detect_weak_areas(req)

        # Generate Recommendations
        recommendations = self._generate_recommendations(req)

        # Generate Weekly Study Schedule
        weekly_plan = self._generate_weekly_plan(req)

        # Generate Plotly Chart Data
        habit_radar = self._generate_radar_data(req)
        study_vs_perf = self._generate_study_vs_performance(req)
        weekly_pattern = self._generate_weekly_pattern(req)

        return PredictionResponse(
            predicted_gpa=predicted_gpa,
            performance_level=performance_level,
            confidence=confidence,
            weak_areas=weak_areas,
            recommendations=recommendations,
            weekly_plan=weekly_plan,
            habit_radar_data=habit_radar,
            study_vs_performance_chart=study_vs_perf,
            weekly_pattern_data=weekly_pattern,
            input_summary={
                "study_hours": req.study_hours,
                "sleep_hours": req.sleep_hours,
                "attendance": req.attendance,
                "assignment_completion": req.assignment_completion,
                "previous_gpa": req.previous_gpa,
                "screen_time": req.screen_time,
                "self_study_hours": req.self_study_hours,
                "revision_frequency": req.revision_frequency,
                "test_frequency": req.test_frequency
            }
        )

    def _detect_weak_areas(self, req: PredictionRequest) -> List[WeakAreaItem]:
        weak_areas = []

        if req.study_hours < 3.5 or (req.study_days or 5) < 4:
            weak_areas.append(WeakAreaItem(
                area="Study Consistency",
                problem="Low daily study volume & frequency",
                current_value=f"{req.study_hours} hrs/day ({req.study_days or 5} days/wk)",
                target_value="4.5+ hrs/day (5-6 days/wk)",
                severity="high",
                icon="BookOpen"
            ))

        if req.attendance < 80.0:
            weak_areas.append(WeakAreaItem(
                area="Class Attendance",
                problem="Sub-optimal class attendance percentage",
                current_value=f"{req.attendance}%",
                target_value="85%+ attendance",
                severity="high",
                icon="UserCheck"
            ))

        if req.screen_time > 4.0:
            weak_areas.append(WeakAreaItem(
                area="Screen Time Management",
                problem="Excessive non-academic screen distraction",
                current_value=f"{req.screen_time} hrs/day",
                target_value="< 2.5 hrs/day",
                severity="medium",
                icon="Smartphone"
            ))

        if req.sleep_hours < 6.5 or req.sleep_hours > 9.5:
            weak_areas.append(WeakAreaItem(
                area="Sleep Schedule",
                problem="Irregular sleep pattern affecting retention",
                current_value=f"{req.sleep_hours} hrs/night",
                target_value="7.0 - 8.0 hrs/night",
                severity="medium",
                icon="Moon"
            ))

        if req.revision_frequency < 3:
            weak_areas.append(WeakAreaItem(
                area="Revision Frequency",
                problem="Insufficient periodic review of studied material",
                current_value=f"{req.revision_frequency} days/week",
                target_value="4+ days/week",
                severity="medium",
                icon="RotateCcw"
            ))

        if req.test_frequency < 2:
            weak_areas.append(WeakAreaItem(
                area="Practice & Self-Testing",
                problem="Low practice exam and active recall frequency",
                current_value=f"{req.test_frequency} times/week",
                target_value="3+ times/week",
                severity="low",
                icon="BrainCircuit"
            ))

        if not weak_areas:
            weak_areas.append(WeakAreaItem(
                area="Advanced Mastery",
                problem="Good habit baseline — optimize peak focus sessions",
                current_value="High performance baseline",
                target_value="Maintain focus & mock tests",
                severity="low",
                icon="Award"
            ))

        return weak_areas

    def _generate_recommendations(self, req: PredictionRequest) -> List[RecommendationItem]:
        recs = []
        target_study_hours = max(4.5, req.study_hours + 0.75)
        
        recs.append(RecommendationItem(
            category="Study Plan",
            recommendation=f"Increase focused study time by 45 minutes per day to reach target of {target_study_hours:.1f} hours daily using the Pomodoro technique.",
            priority="High" if req.study_hours < 3.5 else "Medium",
            icon="Clock"
        ))

        if req.sleep_hours < 7.0:
            recs.append(RecommendationItem(
                category="Sleep & Health",
                recommendation="Maintain 7.5 hours of consistent sleep per night. Cognitive memory consolidation peaks during deep REM sleep cycles.",
                priority="High" if req.sleep_hours < 6.0 else "Medium",
                icon="Moon"
            ))
        else:
            recs.append(RecommendationItem(
                category="Sleep & Health",
                recommendation="Great job on sleep duration! Keep maintaining your regular bedtime routine to sustain focus.",
                priority="Low",
                icon="CheckCircle"
            ))

        recs.append(RecommendationItem(
            category="Revision Strategy",
            recommendation=f"Schedule spaced repetition revision sessions at least {max(3, int(req.revision_frequency + 1))} times per week to prevent memory decay.",
            priority="High" if req.revision_frequency < 3 else "Medium",
            icon="Repeat"
        ))

        if req.screen_time > 3.0:
            recs.append(RecommendationItem(
                category="Digital Distractions",
                recommendation=f"Reduce non-academic screen time from {req.screen_time}h to under 2.5h daily by using app blockers during deep study blocks.",
                priority="High" if req.screen_time > 4.5 else "Medium",
                icon="Smartphone"
            ))

        recs.append(RecommendationItem(
            category="Active Recall",
            recommendation="Incorporate active recall self-tests and past year question practice twice a week before exams.",
            priority="Medium",
            icon="FileCheck"
        ))

        return recs

    def _generate_weekly_plan(self, req: PredictionRequest) -> List[DaySchedule]:
        dept = req.department or "Computer Science"
        if "CS" in dept or "Computer" in dept or "IT" in dept or "Tech" in dept:
            subjects = ["Data Structures & Algo", "Database Systems", "Operating Systems", "Computer Networks", "Software Engineering"]
        elif "Electr" in dept or "ECE" in dept or "EEE" in dept:
            subjects = ["Circuit Analysis", "Digital Signals", "Microprocessors", "Electromagnetics", "Control Systems"]
        elif "Mech" in dept or "Civil" in dept:
            subjects = ["Thermodynamics", "Fluid Mechanics", "Strength of Materials", "Engineering Math", "CAD Design"]
        else:
            subjects = ["Core Subject 1", "Core Subject 2", "Applied Mathematics", "Lab & Practical", "Elective Specialization"]

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        plan = []
        base_hours = max(2.5, min(6.0, req.study_hours))

        for i, day in enumerate(days):
            subj = subjects[i % len(subjects)]
            if day in ["Saturday", "Sunday"]:
                day_hours = round(base_hours * 1.2, 1)
                rev_topic = f"Weekly Mock Test & Intensive Review on {subjects[(i+1)%len(subjects)]}"
                test_str = "Full Length Mock Quiz (60 mins)"
            else:
                day_hours = round(base_hours, 1)
                rev_topic = f"Spaced Revision: {subj} Key Formulas & Notes"
                test_str = "Flashcards & Practice Questions (30 mins)"

            plan.append(DaySchedule(
                day=day,
                study_hours=day_hours,
                subject=subj,
                revision=rev_topic,
                test_practice=test_str
            ))

        return plan

    def _generate_radar_data(self, req: PredictionRequest):
        return [
            {"subject": "Study Hours", "Student": round(min(100.0, (req.study_hours / 6.0) * 100), 1), "Benchmark": 85},
            {"subject": "Sleep", "Student": round(min(100.0, (req.sleep_hours / 8.0) * 100), 1), "Benchmark": 90},
            {"subject": "Attendance", "Student": round(req.attendance, 1), "Benchmark": 90},
            {"subject": "Assignments", "Student": round(req.assignment_completion, 1), "Benchmark": 95},
            {"subject": "Revision", "Student": round(min(100.0, (req.revision_frequency / 5.0) * 100), 1), "Benchmark": 80},
            {"subject": "Tests & Practice", "Student": round(min(100.0, (req.test_frequency / 4.0) * 100), 1), "Benchmark": 80}
        ]

    def _generate_study_vs_performance(self, req: PredictionRequest):
        curve = []
        for h in range(1, 11):
            sample = pd.DataFrame([{
                "study_hours": h,
                "sleep_hours": req.sleep_hours,
                "attendance": req.attendance,
                "assignment_completion": req.assignment_completion,
                "previous_gpa": req.previous_gpa,
                "screen_time": req.screen_time,
                "self_study_hours": req.self_study_hours,
                "revision_frequency": req.revision_frequency,
                "test_frequency": req.test_frequency,
                "exercise_hours": req.exercise_hours or 3.0,
                "class_participation": req.class_participation or 7.0
            }])[self.feature_names]

            pred_gpa = round(float(np.clip(self.model.predict(sample)[0], 0.0, 10.0)), 2)
            curve.append({
                "study_hours": f"{h} hrs",
                "predicted_gpa": pred_gpa,
                "is_current": (round(req.study_hours) == h)
            })
        return curve

    def _generate_weekly_pattern(self, req: PredictionRequest):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hours_per_day = [
            round(req.study_hours * 0.9, 1),
            round(req.study_hours * 1.0, 1),
            round(req.study_hours * 1.1, 1),
            round(req.study_hours * 0.95, 1),
            round(req.study_hours * 0.85, 1),
            round(req.study_hours * 1.25, 1),
            round(req.study_hours * 1.15, 1)
        ]
        return [{"day": d, "hours": h} for d, h in zip(days, hours_per_day)]

predictor_instance = AcademicPerformancePredictor()
