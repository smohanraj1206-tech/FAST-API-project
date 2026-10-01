import requests
import numpy as np

class FastAPIClient:
    """
    HTTP Client service for Streamlit to interact with FastAPI REST API endpoints.
    Includes offline client-side demo mode fallback.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> tuple[bool, dict]:
        """
        Pings GET /health endpoint
        """
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            if r.status_code == 200:
                return True, r.json()
            return False, {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            return False, {"error": str(e)}

    def predict_performance(self, payload: dict, is_demo: bool = False) -> tuple[bool, dict]:
        """
        Posts study habits payload to POST /predict endpoint.
        Falls back to demo engine if is_demo is True.
        """
        if is_demo:
            return True, self.generate_demo_prediction(payload)

        try:
            r = requests.post(f"{self.base_url}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                return True, r.json()
            return False, {"error": f"HTTP {r.status_code}: {r.text}"}
        except Exception as e:
            return False, {"error": f"Unable to connect to FastAPI server at {self.base_url}. Error: {str(e)}"}

    @staticmethod
    def generate_demo_prediction(inputs: dict) -> dict:
        """
        Offline client-side prediction calculation engine.
        """
        sh = float(inputs.get("study_hours", 4.5))
        sl = float(inputs.get("sleep_hours", 7.0))
        att = float(inputs.get("attendance", 85.0))
        ass = float(inputs.get("assignment_completion", 90.0))
        prevGpa = float(inputs.get("previous_gpa", 7.8))
        scr = float(inputs.get("screen_time", 3.0))
        rev = float(inputs.get("revision_frequency", 4.0))
        tst = float(inputs.get("test_frequency", 2.0))

        sleepFactor = -0.08 * ((sl - 7.5)**2) + 0.3
        screenPenalty = -0.12 * max(0, scr - 3.0)

        gpa = 0.35 * prevGpa + 0.20 * (sh * 0.8) + 0.025 * att + 0.018 * ass + 0.12 * rev + 0.10 * tst + sleepFactor + screenPenalty
        gpa = round(float(np.clip(gpa, 2.0, 10.0)), 2)

        level = "Good"
        if gpa >= 8.5: level = "Excellent"
        elif gpa >= 7.2: level = "Good"
        elif gpa >= 6.0: level = "Average"
        else: level = "Needs Improvement"

        weak_areas = []
        if sh < 3.5:
            weak_areas.append({"area": "Study Consistency", "problem": "Low daily study volume", "current_value": f"{sh} hrs/day", "target_value": "4.5+ hrs/day", "severity": "high", "icon": "BookOpen"})
        if att < 80:
            weak_areas.append({"area": "Class Attendance", "problem": "Sub-optimal attendance percentage", "current_value": f"{att}%", "target_value": "85%+", "severity": "high", "icon": "UserCheck"})
        if scr > 4:
            weak_areas.append({"area": "Screen Time", "problem": "High non-academic screen usage", "current_value": f"{scr} hrs/day", "target_value": "< 2.5 hrs/day", "severity": "medium", "icon": "Smartphone"})
        if sl < 6.5:
            weak_areas.append({"area": "Sleep Schedule", "problem": "Rest period below optimal baseline", "current_value": f"{sl} hrs/night", "target_value": "7–8 hrs/night", "severity": "medium", "icon": "Moon"})
        if rev < 3:
            weak_areas.append({"area": "Revision Frequency", "problem": "Low review frequency leading to retention drop", "current_value": f"{rev} days/week", "target_value": "4+ days/week", "severity": "medium", "icon": "RotateCcw"})
        if not weak_areas:
            weak_areas.append({"area": "Advanced Mastery", "problem": "Good baseline — maintain focus", "current_value": "Optimal baseline", "target_value": "Maintain mock drills", "severity": "low", "icon": "Award"})

        recs = [
            {"category": "Study Plan", "recommendation": f"Increase focused study time by 45 minutes per day to reach target of {max(4.5, sh + 0.75):.1f} hours daily.", "priority": "High" if sh < 3.5 else "Medium", "icon": "Clock"},
            {"category": "Sleep & Health", "recommendation": "Maintain 7.5 hours of consistent sleep per night for cognitive retention.", "priority": "High" if sl < 6.5 else "Low", "icon": "Moon"},
            {"category": "Revision Strategy", "recommendation": "Schedule spaced repetition revision sessions at least 4 times per week.", "priority": "High" if rev < 3 else "Medium", "icon": "Repeat"},
            {"category": "Digital Distractions", "recommendation": "Reduce non-academic screen time during study sessions.", "priority": "High" if scr > 4 else "Low", "icon": "Smartphone"}
        ]

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_plan = [{"day": d, "study_hours": round(sh * 1.2, 1) if d in ["Saturday", "Sunday"] else round(sh, 1), "subject": f"Core Subject {(i%4)+1}", "revision": f"Review Chapter {(i%5)+1} notes", "test_practice": "Practice Drill (30 mins)"} for i, d in enumerate(days)]

        radar_data = [
            {"subject": "Study", "Student": min(100, int((sh/6.0)*100)), "Benchmark": 85},
            {"subject": "Sleep", "Student": min(100, int((sl/8.0)*100)), "Benchmark": 90},
            {"subject": "Attendance", "Student": int(att), "Benchmark": 90},
            {"subject": "Assignments", "Student": int(ass), "Benchmark": 95},
            {"subject": "Revision", "Student": min(100, int((rev/5.0)*100)), "Benchmark": 80},
            {"subject": "Tests", "Student": min(100, int((tst/4.0)*100)), "Benchmark": 80}
        ]

        curve = [{"study_hours": f"{h} hrs", "predicted_gpa": round(float(np.clip(0.35 * prevGpa + 0.20 * (h * 0.8) + 0.025 * att + 0.018 * ass + 0.12 * rev + sleepFactor + screenPenalty, 2.0, 10.0)), 2)} for h in range(1, 11)]

        weekly_pattern = [{"day": d, "hours": [round(sh*0.9, 1), round(sh, 1), round(sh*1.1, 1), round(sh*0.95, 1), round(sh*0.85, 1), round(sh*1.25, 1), round(sh*1.15, 1)][i]} for i, d in enumerate(days)]

        return {
            "predicted_gpa": gpa,
            "performance_level": level,
            "confidence": 88.0,
            "weak_areas": weak_areas,
            "recommendations": recs,
            "weekly_plan": weekly_plan,
            "habit_radar_data": radar_data,
            "study_vs_performance_chart": curve,
            "weekly_pattern_data": weekly_pattern,
            "input_summary": inputs
        }
