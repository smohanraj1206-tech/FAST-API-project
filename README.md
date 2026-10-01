# AI-Powered Study Habit Analyzer & Performance Predictor

An end-to-end artificial intelligence web application designed for college students to analyze study habits, predict academic performance (GPA), identify weak areas, and receive personalized study recommendations using Machine Learning, FastAPI, and Streamlit.

---

## 🏗️ Architecture & Technology Stack

```text
┌─────────────────────────┐        HTTP POST        ┌─────────────────────────┐
│  Streamlit Frontend UI  │  ─────────────────────► │   FastAPI Backend API   │
│   (streamlit_app.py)    │ ◄─────────────────────  │      (api/main.py)      │
└─────────────────────────┘      JSON Response      └────────────┬────────────┘
                                                                     │
                                                                     ▼
                                                        ┌─────────────────────────┐
                                                        │ Scikit-Learn ML Model   │
                                                        │ (model/train.py &       │
                                                        │  model/predict.py)      │
                                                        └─────────────────────────┘
```

- **Frontend Dashboard**: Streamlit (Python UI components, Plotly Charts)
- **Backend API**: FastAPI (REST architecture, Pydantic validation, CORS middleware)
- **Machine Learning Engine**: Scikit-Learn `RandomForestRegressor` and `StandardScaler` pipeline trained on student study habit metrics
- **Dataset**: 2,000 synthetic college student samples (`data/student_performance.csv`)
- **HTTP Client**: `utils.api_client.FastAPIClient` (Python `requests` client)

---

## 📁 Project Structure

```text
AI-Study-Habit-Analyzer/
│
├── streamlit_app.py          # Streamlit frontend application
│
├── api/                      # FastAPI Backend module
│   ├── __init__.py
│   ├── main.py               # FastAPI entry point & CORS configuration
│   ├── routes.py             # APIRouter defining GET /health, POST /predict, POST /analyze-habits
│   └── schemas.py            # Pydantic data schemas
│
├── model/                    # Machine Learning module
│   ├── __init__.py
│   ├── train.py              # Script to train Scikit-Learn model on dataset & save model.pkl
│   ├── predict.py            # AcademicPerformancePredictor inference engine
│   └── model.pkl             # Trained Scikit-Learn model binary file
│
├── data/                     # Dataset directory
│   └── student_performance.csv# 2,000 college student study habit samples
│
├── utils/                    # Utility & API Client module
│   ├── __init__.py
│   └── api_client.py         # FastAPIClient helper for Streamlit HTTP communications
│
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start Instructions

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

### **2. Start the FastAPI Backend**
Run the FastAPI backend server on port 8000:
```bash
uvicorn api.main:app --reload
```
- **Backend Base URL**: `http://localhost:8000`
- **Health Check Endpoint**: `http://localhost:8000/health`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`

---

### **3. Start the Streamlit Frontend**
In a separate terminal window, launch the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```
- **Streamlit Web Dashboard**: `http://localhost:8501`

---

## ⚙️ Features & Pages

1. **🏠 Home**: Overview of the AI prediction platform and core features.
2. **🧠 Analyze Habits**: Input form for student background, daily study metrics, sleep, attendance %, assignment completion, previous GPA, and learning habits. Includes an *"Auto-Fill Sample Data"* button for 1-click testing.
3. **📊 Dashboard**:
   - **Large Prediction Banner**: Predicted GPA (e.g. `8.20 GPA`), Performance Band (`Good`/`Excellent`), ML Model Confidence score (`87.5%`).
   - **Performance Level Meter**: Plotly Indicator Gauge displaying Poor → Average → Good → Excellent with needle marker.
   - **Habit Metrics Cards**: Submitted stat counters with optimal/warning status.
   - **Interactive Plotly Charts**:
     - *Study vs Performance Curve*: Area chart showing GPA trend across study hours.
     - *Habit Distribution Radar*: Radar chart comparing Student profile vs Benchmark.
     - *Weekly Study Pattern*: Bar chart showing daily study hours.
   - **Weak Areas Section**: Identified focus areas with current values vs target metrics.
4. **💡 Recommendations**: Categorized AI study recommendations (*Study Plan*, *Sleep*, *Revision*, *Digital Distractions*) with priority badges and a **7-Day Personalized Study Schedule Planner** grid.
5. **ℹ️ About & Architecture**: Pipeline flowchart, tech stack summary, and academic non-improvement disclaimer.

---

## 🤖 Machine Learning Model Details

- **Algorithm**: Scikit-Learn `RandomForestRegressor` with 100 decision trees and `StandardScaler` feature scaling.
- **Input Features**: `study_hours`, `sleep_hours`, `attendance`, `assignment_completion`, `previous_gpa`, `screen_time`, `self_study_hours`, `revision_frequency`, `test_frequency`, `exercise_hours`, `class_participation`.
- **Target Variable**: Expected GPA (`target_gpa`, scale 0.0 - 10.0).
- **Training Script**: Run `python model/train.py` to retrain the model on updated dataset files.

---

## 📡 API Endpoint Specifications

### `GET /health`
Returns system status and model load state.

### `POST /predict`
**Sample Request Payload:**
```json
{
  "study_hours": 4.5,
  "sleep_hours": 7.0,
  "attendance": 85.0,
  "assignment_completion": 90.0,
  "previous_gpa": 7.8,
  "screen_time": 3.0,
  "self_study_hours": 3.0,
  "revision_frequency": 4.0,
  "test_frequency": 2.0,
  "student_name": "Alex Johnson",
  "age": 20,
  "academic_year": "Junior",
  "department": "Computer Science"
}
```

---

## 🎮 Demo Mode (Offline Testing)

If the FastAPI backend server is offline:
1. Open the sidebar in Streamlit.
2. Check **"Enable Client Demo Mode (Offline)"**.
3. The app will utilize an offline calculation engine to generate predictions, Plotly charts, and study schedules locally.
