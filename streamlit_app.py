import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.api_client import FastAPIClient

# Page Configuration
st.set_page_config(
    page_title="StudyAI | AI Study Habit Analyzer & Performance Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Modern Dashboard Aesthetic
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stCard {
        background-color: #ffffff;
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5 0%, #3730a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #475569;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .prediction-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px -4px rgba(79, 70, 229, 0.25);
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-excellent { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4); }
    .badge-good { background-color: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(165, 180, 252, 0.4); }
    .badge-average { background-color: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(252, 211, 77, 0.4); }
    .badge-poor { background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(252, 165, 165, 0.4); }
    .stButton>button {
        border-radius: 0.75rem;
        font-weight: 700;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'api_url' not in st.session_state:
    st.session_state['api_url'] = 'http://localhost:8000'

if 'is_demo' not in st.session_state:
    st.session_state['is_demo'] = False

if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "🏠 Home"

# Create API Client Instance
api_client = FastAPIClient(base_url=st.session_state['api_url'])

# Default sample result
if st.session_state['prediction_result'] is None:
    st.session_state['prediction_result'] = api_client.generate_demo_prediction({
        'study_hours': 4.5,
        'sleep_hours': 7.0,
        'attendance': 85.0,
        'assignment_completion': 90.0,
        'previous_gpa': 7.8,
        'screen_time': 3.0,
        'self_study_hours': 3.0,
        'revision_frequency': 4.0,
        'test_frequency': 2.0,
        'student_name': 'Alex Johnson',
        'department': 'Computer Science'
    })


# SIDEBAR NAVIGATION & SETTINGS
with st.sidebar:
    st.markdown("## 🧠 StudyAI")
    st.caption("AI-Powered Habit Analyzer & Performance Predictor")
    st.divider()

    nav_options = ["🏠 Home", "🧠 Analyze Habits", "📊 Dashboard", "💡 Recommendations", "ℹ️ About & Architecture"]
    
    try:
        start_idx = nav_options.index(st.session_state['current_page'])
    except ValueError:
        start_idx = 0

    page_selection = st.radio(
        "Navigation",
        nav_options,
        index=start_idx
    )
    st.session_state['current_page'] = page_selection

    st.divider()

    st.markdown("### ⚙️ FastAPI Connection Settings")
    api_input = st.text_input("FastAPI Base URL", value=st.session_state['api_url'])
    if api_input != st.session_state['api_url']:
        st.session_state['api_url'] = api_input
        api_client = FastAPIClient(base_url=api_input)

    # Health Check via API Client
    is_online, health_data = api_client.check_health()

    if is_online:
        st.success(f"🟢 Connected to FastAPI ({st.session_state['api_url']})")
    else:
        st.error(f"🔴 Backend Offline ({st.session_state['api_url']})")

    demo_mode_toggle = st.checkbox("Enable Client Demo Mode (Offline)", value=st.session_state['is_demo'])
    st.session_state['is_demo'] = demo_mode_toggle

    if st.session_state['is_demo']:
        st.warning("⚡ Demo Mode Active: Bypassing FastAPI network call")

    st.divider()
    st.caption("v1.0 ML | Streamlit + FastAPI + Scikit-Learn")


# PAGE 1: HOME
if page_selection == "🏠 Home":
    st.markdown('<div class="main-title">Understand Your Study Habits. Improve Your Performance.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyze your study patterns, predict your academic performance, and receive personalized recommendations powered by Machine Learning.</div>', unsafe_allow_html=True)

    col_hero_1, col_hero_2 = st.columns([7, 5])
    
    with col_hero_1:
        st.info("💡 **Scikit-Learn ML Model & FastAPI Integration Ready**")
        st.markdown("""
        Our machine learning model analyzes key academic dimensions—such as study hours, attendance rate, assignment completion velocity, sleep patterns, and revision frequency—to provide an accurate forecast of your expected GPA.
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Analyze My Habits", type="primary"):
                st.session_state['current_page'] = "🧠 Analyze Habits"
                st.rerun()
        with c2:
            if st.button("📊 View Performance Dashboard"):
                st.session_state['current_page'] = "📊 Dashboard"
                st.rerun()

    with col_hero_2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e1b4b, #4338ca); padding: 1.5rem; border-radius: 1rem; color: white;">
            <h4 style="margin:0; color: #a5b4fc;">ML Inference Preview</h4>
            <h2 style="margin:0.5rem 0; color: #fcd34d;">Predicted: 8.45 GPA</h2>
            <p style="font-size: 0.85rem; color: #e0e7ff;">Model Confidence: <b>89%</b> | Status: <b>Optimal Baseline</b></p>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <p style="font-size: 0.8rem; color: #c7d2fe; margin:0;">AI Insight: Increasing revision frequency from 2 to 4 days/week boosts predicted GPA by +0.65 points.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌟 Four Core Intelligence Features")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown("#### 📖 Study Habit Analysis")
        st.caption("Analyze daily study patterns, sleep hours, attendance %, and screen time velocity.")

    with f2:
        st.markdown("#### 🎯 Performance Prediction")
        st.caption("Predict expected academic performance using Machine Learning models trained on student metrics.")

    with f3:
        st.markdown("#### ⚠️ Weak Area Detection")
        st.caption("Identify habit bottlenecks with automated current vs recommended target benchmarks.")

    with f4:
        st.markdown("#### 📅 Personalized Recommendations")
        st.caption("Generate actionable study recommendations and a customized 7-day study planner.")


# PAGE 2: ANALYZE HABITS
elif page_selection == "🧠 Analyze Habits":
    st.markdown('<div class="main-title">Analyze Your Study Habits</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Input your academic metrics to query the FastAPI POST /predict Machine Learning endpoint.</div>', unsafe_allow_html=True)

    if st.button("🪄 Auto-Fill High-Performing Sample Data"):
        st.session_state['autofill'] = True
    else:
        if 'autofill' not in st.session_state:
            st.session_state['autofill'] = False

    autofill = st.session_state['autofill']

    with st.form("habits_analysis_form"):
        
        st.markdown("### 1️⃣ Student Profile Information")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            name_in = st.text_input("Student Name", value="Jordan Miller" if autofill else "Alex Johnson")
        with col_s2:
            age_in = st.number_input("Age", min_value=15, max_value=60, value=21 if autofill else 20)
        with col_s3:
            year_in = st.selectbox("Academic Year", ["Freshman", "Sophomore", "Junior", "Senior"], index=2)
        with col_s4:
            dept_in = st.text_input("Department / Major", value="Computer Science & AI" if autofill else "Computer Science")

        st.markdown("---")
        st.markdown("### 2️⃣ Study & Lifestyle Metrics")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            study_hrs_in = st.slider("Average Study Hours Per Day", min_value=0.0, max_value=14.0, value=5.0 if autofill else 4.5, step=0.5)
            sleep_hrs_in = st.slider("Average Sleep Hours Per Night", min_value=4.0, max_value=12.0, value=7.5 if autofill else 7.0, step=0.5)
            attendance_in = st.slider("Class Attendance Percentage (%)", min_value=40.0, max_value=100.0, value=92.0 if autofill else 85.0, step=1.0)
            assignment_in = st.slider("Assignment Completion Percentage (%)", min_value=30.0, max_value=100.0, value=95.0 if autofill else 90.0, step=1.0)

        with col_m2:
            prev_gpa_in = st.number_input("Previous Semester GPA (0.0 - 10.0 Scale)", min_value=0.0, max_value=10.0, value=8.4 if autofill else 7.8, step=0.1)
            screen_time_in = st.number_input("Average Daily Non-Academic Screen Time (hrs)", min_value=0.0, max_value=18.0, value=2.5 if autofill else 3.0, step=0.5)
            study_days_in = st.selectbox("Study Days Per Week", [1, 2, 3, 4, 5, 6, 7], index=5 if autofill else 4)
            exercise_in = st.number_input("Exercise Hours Per Week", min_value=0.0, max_value=24.0, value=5.0 if autofill else 3.0, step=0.5)

        st.markdown("---")
        st.markdown("### 3️⃣ Learning & Revision Habits")
        
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            self_study_in = st.number_input("Self Study Hours / Day", min_value=0.0, max_value=12.0, value=3.5 if autofill else 3.0, step=0.5)
        with col_l2:
            partic_in = st.selectbox("Class Participation Score (1-10)", list(range(1, 11)), index=8 if autofill else 7)
        with col_l3:
            revision_in = st.selectbox("Revision Frequency (Days/Week)", list(range(0, 8)), index=5 if autofill else 4)
        with col_l4:
            test_in = st.selectbox("Practice Drills / Week", list(range(0, 8)), index=3 if autofill else 2)

        submit_btn = st.form_submit_button("⚡ Analyze My Performance", type="primary")

    if submit_btn:
        payload = {
            "study_hours": study_hrs_in,
            "sleep_hours": sleep_hrs_in,
            "attendance": attendance_in,
            "assignment_completion": assignment_in,
            "previous_gpa": prev_gpa_in,
            "screen_time": screen_time_in,
            "self_study_hours": self_study_in,
            "revision_frequency": revision_in,
            "test_frequency": test_in,
            "student_name": name_in,
            "age": age_in,
            "academic_year": year_in,
            "department": dept_in,
            "study_days": study_days_in,
            "exercise_hours": exercise_in,
            "class_participation": partic_in
        }

        with st.spinner("Connecting to FastAPI backend & calculating ML prediction..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.002)
                progress_bar.progress(i + 1)

            success, result_data = api_client.predict_performance(payload, is_demo=st.session_state['is_demo'])

            if success:
                st.session_state['prediction_result'] = result_data
                st.success("Analysis complete!")
                time.sleep(0.2)
                st.session_state['current_page'] = "📊 Dashboard"
                st.rerun()
            else:
                st.error(result_data.get("error", "Error connecting to FastAPI backend"))
                st.info("💡 You can switch to **Demo Mode (Offline)** in the sidebar to test the UI.")


# PAGE 3: DASHBOARD
elif page_selection == "📊 Dashboard":
    st.markdown('<div class="main-title">Your Performance Analysis Dashboard</div>', unsafe_allow_html=True)
    
    res = st.session_state['prediction_result']
    
    if not res:
        st.warning("No prediction data available yet. Please complete the form on the **Analyze Habits** page.")
        if st.button("Go to Analyze Page"):
            st.session_state['current_page'] = "🧠 Analyze Habits"
            st.rerun()
    else:
        gpa = res.get('predicted_gpa', 8.2)
        level = res.get('performance_level', 'Good')
        confidence = res.get('confidence', 87.5)

        badge_cls = "badge-excellent" if level=="Excellent" else "badge-good" if level=="Good" else "badge-average" if level=="Average" else "badge-poor"
        
        st.markdown(f"""
        <div class="prediction-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <span class="badge-pill {badge_cls}">AI Prediction Target</span>
                    <h1 style="margin: 0.5rem 0; font-size: 2.8rem; font-weight:800; color: #ffffff;">
                        Predicted Performance: <span style="color: #fcd34d;">{gpa} GPA</span>
                    </h1>
                    <p style="color: #cbd5e1; margin:0; font-size: 0.95rem;">
                        Processed via FastAPI ML Model Pipeline | Performance Band: <b>{level}</b> | ML Confidence: <b>{confidence}%</b>
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_gauge, col_stats = st.columns([5, 7])

        with col_gauge:
            st.markdown("### 🎛️ Performance Level Meter")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = gpa,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Band: {level}", 'font': {'size': 18, 'color': '#1e293b'}},
                gauge = {
                    'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#4f46e5"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 6.0], 'color': '#fca5a5'},
                        {'range': [6.0, 7.2], 'color': '#fde68a'},
                        {'range': [7.2, 8.5], 'color': '#c7d2fe'},
                        {'range': [8.5, 10.0], 'color': '#6ee7b7'}
                    ],
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_stats:
            st.markdown("### 📌 Submitted Habit Metrics")
            inputs = res.get('input_summary', {})
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Study Hours", f"{inputs.get('study_hours', '--')} h/day")
                st.metric("Assignment Completion", f"{inputs.get('assignment_completion', '--')}%")
            with m2:
                st.metric("Sleep Hours", f"{inputs.get('sleep_hours', '--')} h/night")
                st.metric("Previous GPA", f"{inputs.get('previous_gpa', '--')}")
            with m3:
                st.metric("Class Attendance", f"{inputs.get('attendance', '--')}%")
                st.metric("Daily Screen Time", f"{inputs.get('screen_time', '--')} h/day")

        st.markdown("---")
        st.markdown("### 📈 Interactive Academic Analytics Charts")

        tab_curve, tab_radar, tab_weekly = st.tabs(["📉 Study vs Performance Curve", "🕸️ Habit Distribution Radar", "📊 Weekly Study Pattern"])

        with tab_curve:
            curve_data = res.get('study_vs_performance_chart', [])
            if curve_data:
                df_curve = pd.DataFrame(curve_data)
                fig_curve = px.area(
                    df_curve, 
                    x="study_hours", 
                    y="predicted_gpa", 
                    title="Predicted GPA Trend vs Daily Study Hours",
                    labels={"study_hours": "Study Hours per Day", "predicted_gpa": "Predicted GPA (0 - 10)"},
                    color_discrete_sequence=["#4f46e5"]
                )
                fig_curve.update_layout(height=320, hovermode="x unified")
                st.plotly_chart(fig_curve, use_container_width=True)
            else:
                st.info("Study vs performance curve data is unavailable.")

        with tab_radar:
            radar_data = res.get('habit_radar_data', [])
            if radar_data:
                df_radar = pd.DataFrame(radar_data)
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=df_radar['Student'],
                    theta=df_radar['subject'],
                    fill='toself',
                    name='Student Profile',
                    line_color='#4f46e5'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=df_radar['Benchmark'],
                    theta=df_radar['subject'],
                    fill='toself',
                    name='Optimal Benchmark',
                    line_color='#10b981'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    height=350
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Habit radar distribution data is unavailable.")

        with tab_weekly:
            weekly_data = res.get('weekly_pattern_data', [])
            if weekly_data:
                df_weekly = pd.DataFrame(weekly_data)
                fig_weekly = px.bar(
                    df_weekly, 
                    x="day", 
                    y="hours", 
                    title="Weekly Study Hours Pattern (Mon - Sun)",
                    labels={"day": "Day of Week", "hours": "Study Hours"},
                    color_discrete_sequence=["#6366f1"]
                )
                fig_weekly.update_layout(height=320)
                st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.warning("⚠️ Weekly study pattern data is not available from the backend.")

        st.markdown("---")
        st.markdown("### ⚠️ Areas That Need Attention")
        
        weak_areas = res.get('weak_areas', [])
        if weak_areas:
            w_cols = st.columns(min(3, len(weak_areas)))
            for idx, item in enumerate(weak_areas):
                with w_cols[idx % len(w_cols)]:
                    st.error(f"**{item.get('area', 'Focus Area')}**")
                    st.write(f"**Problem:** {item.get('problem')}")
                    st.caption(f"Current: `{item.get('current_value')}` → Target: `{item.get('target_value')}`")
        else:
            st.success("No critical weak areas detected! Excellent habit baseline.")


# PAGE 4: RECOMMENDATIONS
elif page_selection == "💡 Recommendations":
    st.markdown('<div class="main-title">Personalized Study Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Actionable AI-generated study recommendations and customized 7-day planner.</div>', unsafe_allow_html=True)

    res = st.session_state['prediction_result']
    if not res:
        st.warning("Please submit your study habits on the **Analyze Habits** page first.")
    else:
        st.markdown("### 📋 AI Category Recommendations")
        recs = res.get('recommendations', [])

        r_cols = st.columns(2)
        for idx, rec in enumerate(recs):
            with r_cols[idx % 2]:
                prio = rec.get('priority', 'Medium')
                prio_color = "red" if prio=="High" else "orange" if prio=="Medium" else "green"
                
                st.markdown(f"""
                <div style="background-color: white; border-radius: 0.75rem; padding: 1.25rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#1e293b;">{rec.get('category')}</h4>
                        <span style="font-size:0.75rem; font-weight:bold; color:{prio_color}; border:1px solid {prio_color}; padding:0.1rem 0.5rem; border-radius:0.5rem;">
                            {prio} Priority
                        </span>
                    </div>
                    <p style="color:#475569; font-size:0.9rem; margin-top:0.75rem;">
                        {rec.get('recommendation')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🗓️ 7-Day Personalized Study Schedule Planner")
        
        plan = res.get('weekly_plan', [])
        if plan:
            df_plan = pd.DataFrame(plan)
            df_plan.columns = ["Day", "Study Hours", "Subject Focus", "Revision Tasks", "Practice Drills"]
            st.dataframe(
                df_plan,
                use_container_width=True,
                column_config={
                    "Study Hours": st.column_config.NumberColumn(format="%.1f hrs")
                },
                hide_index=True
            )
        else:
            st.info("No weekly schedule available.")


# PAGE 5: ABOUT & ARCHITECTURE
elif page_selection == "ℹ️ About & Architecture":
    st.markdown('<div class="main-title">About the Platform & Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 System Purpose
    The **AI-Powered Study Habit Analyzer & Performance Predictor** helps college students understand how daily routines—such as study hours, sleep, class attendance, and revision frequency—directly impact predicted GPA.
    
    ---
    
    ### 🏗️ End-to-End Architecture
    ```
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

    ---

    ### 🛠️ Technology Breakdown
    * **Frontend UI**: Streamlit (Python UI components, Plotly Charts)
    * **API Client**: `utils.api_client.FastAPIClient` (Python `requests` client)
    * **Backend REST Server**: FastAPI (`api/main.py` + `api/routes.py` + `api/schemas.py`)
    * **Machine Learning**: Scikit-Learn (`model/train.py` & `model/predict.py` using `model/model.pkl`)
    * **Dataset**: `data/student_performance.csv` (2,000 synthetic college student samples)

    ---

    ### ⚠️ Academic Disclaimer
    This application provides statistical machine learning estimates based on study habit parameters. It does NOT guarantee specific examination scores or final GPA outcomes.
    """)
