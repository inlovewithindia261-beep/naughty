"""
Employee Attrition Prediction - Streamlit App
AIML Summer Internship 2026, IIHMF, MNNIT Allahabad
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Artifacts ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'Model')

@st.cache_resource
def load_artifacts():
    model       = joblib.load(f'{MODEL_DIR}/best_model.pkl')
    scaler      = joblib.load(f'{MODEL_DIR}/scaler.pkl')
    features    = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
    le_dict     = joblib.load(f'{MODEL_DIR}/label_encoders.pkl')
    feat_info   = joblib.load(f'{MODEL_DIR}/feature_info.pkl')
    return model, scaler, features, le_dict, feat_info

model, scaler, top_features, le_dict, feat_info = load_artifacts()
best_model_name = feat_info['best_model_name']

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 25px;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8dadc; font-size: 1rem; margin: 5px 0 0 0; }
    .result-card {
        padding: 20px; border-radius: 10px; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 15px 0;
    }
    .result-leave { background: #fdecea; color: #c0392b; border: 2px solid #e74c3c; }
    .result-stay  { background: #eafaf1; color: #1e8449; border: 2px solid #27ae60; }
    .metric-box {
        background: #f8f9fa; border-radius: 8px; padding: 15px;
        text-align: center; border-left: 4px solid #0f3460;
    }
    .section-title { color: #0f3460; font-size: 1.2rem; font-weight: bold; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>👥 Employee Attrition Prediction System</h1>
    <p>AIML Summer Internship 2026 | IIHMF, MNNIT Allahabad | Capstone Project 4</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar Inputs ──────────────────────────────────────────────
st.sidebar.markdown("## 📋 Employee Details")
st.sidebar.markdown("Fill in the employee's information below:")

with st.sidebar:
    st.markdown("### 👤 Personal Information")
    age            = st.slider("Age", 18, 60, 35)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education      = st.selectbox("Education Level",
                                  [1, 2, 3, 4, 5],
                                  format_func=lambda x: {
                                      1:"Below College",2:"College",
                                      3:"Bachelor",4:"Master",5:"Doctor"}[x])
    education_field = st.selectbox("Education Field",
                                   ["Life Sciences","Medical","Marketing",
                                    "Technical Degree","Human Resources","Other"])

    st.markdown("### 🏢 Job Information")
    department     = st.selectbox("Department",
                                  ["Sales","Research & Development","Human Resources"])
    job_role       = st.selectbox("Job Role",
                                  ["Sales Executive","Research Scientist",
                                   "Laboratory Technician","Manager","Sales Representative",
                                   "Research Director","Human Resources"])
    job_level      = st.selectbox("Job Level", [1,2,3,4,5])
    job_involvement= st.slider("Job Involvement (1-Low, 4-High)", 1, 4, 3)
    overtime       = st.selectbox("OverTime", ["Yes", "No"])
    business_travel= st.selectbox("Business Travel",
                                  ["Non-Travel","Travel_Rarely","Travel_Frequently"])

    st.markdown("### 💰 Compensation")
    monthly_income = st.number_input("Monthly Income ($)", 1009, 20000, 5000, 500)
    percent_salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 15)
    stock_option_level  = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    daily_rate     = st.number_input("Daily Rate ($)", 100, 1500, 800, 50)

    st.markdown("### 📊 Satisfaction Scores")
    job_satisfaction     = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    env_satisfaction     = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    relationship_sat     = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
    work_life_balance    = st.slider("Work-Life Balance (1-4)", 1, 4, 3)

    st.markdown("### 📅 Experience & Tenure")
    total_working_years  = st.slider("Total Working Years", 0, 40, 10)
    years_at_company     = st.slider("Years at Company", 0, 40, 5)
    years_in_role        = st.slider("Years in Current Role", 0, 18, 3)
    years_with_manager   = st.slider("Years with Current Manager", 0, 17, 3)
    years_since_promo    = st.slider("Years Since Last Promotion", 0, 15, 2)
    num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)
    training_times       = st.slider("Training Times Last Year", 0, 6, 3)
    distance_from_home   = st.slider("Distance From Home (km)", 1, 29, 10)
    performance_rating   = st.selectbox("Performance Rating", [3, 4])

# ─── Feature Engineering (mirror pipeline) ───────────────────────
def encode_cat(col, val):
    le = le_dict.get(col)
    if le and val in le.classes_:
        return le.transform([val])[0]
    return 0

input_dict = {
    'Age': age,
    'DailyRate': daily_rate,
    'DistanceFromHome': distance_from_home,
    'Education': education,
    'EnvironmentSatisfaction': env_satisfaction,
    'HourlyRate': 60,
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobSatisfaction': job_satisfaction,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': 15000,
    'NumCompaniesWorked': num_companies_worked,
    'PercentSalaryHike': percent_salary_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_sat,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': training_times,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_role,
    'YearsSinceLastPromotion': years_since_promo,
    'YearsWithCurrManager': years_with_manager,
    # encoded cats
    'BusinessTravel_enc': encode_cat('BusinessTravel', business_travel),
    'Department_enc': encode_cat('Department', department),
    'EducationField_enc': encode_cat('EducationField', education_field),
    'Gender_enc': encode_cat('Gender', gender),
    'JobRole_enc': encode_cat('JobRole', job_role),
    'MaritalStatus_enc': encode_cat('MaritalStatus', marital_status),
    'OverTime_enc': encode_cat('OverTime', overtime),
    # engineered
    'IncomeToExperience': monthly_income / (total_working_years + 1),
    'SatisfactionIndex': (job_satisfaction + env_satisfaction +
                          relationship_sat + work_life_balance) / 4,
    'TenureRatio': years_in_role / (years_at_company + 1),
    'OverdueProm': int(years_since_promo > 3),
    'IsJunior': int(job_level <= 2),
    'LowIncome': int(monthly_income < 3500),
}

input_df = pd.DataFrame([input_dict])
# Align to expected features
for col in top_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[top_features]
input_scaled = scaler.transform(input_df)

# ─── Prediction ──────────────────────────────────────────────────
prob = model.predict_proba(input_scaled)[0][1]
pred = int(prob >= 0.5)

# ─── Main Panel ──────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🔍 Prediction Result")

    if pred == 1:
        st.markdown(f"""
        <div class="result-card result-leave">
            ⚠️ HIGH ATTRITION RISK — Employee Likely to Leave<br>
            <span style="font-size:1rem; font-weight:normal;">Attrition Probability: <b>{prob:.1%}</b></span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-stay">
            ✅ LOW ATTRITION RISK — Employee Likely to Stay<br>
            <span style="font-size:1rem; font-weight:normal;">Attrition Probability: <b>{prob:.1%}</b></span>
        </div>""", unsafe_allow_html=True)

    # Gauge chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=18, color='#EEEEEE', solid_capstyle='round')
    risk_theta = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    color = '#E74C3C' if prob > 0.5 else ('#F39C12' if prob > 0.3 else '#27AE60')
    ax.plot(np.cos(risk_theta), np.sin(risk_theta), lw=18, color=color, solid_capstyle='round')
    angle = np.pi - prob * np.pi
    ax.annotate('', xy=(0.65 * np.cos(angle), 0.65 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.5))
    ax.text(0, -0.15, f'{prob:.1%}', ha='center', va='center', fontsize=22,
            fontweight='bold', color=color)
    ax.text(0, -0.38, 'Attrition Risk', ha='center', fontsize=10, color='#555')
    ax.text(-1.0, -0.25, 'LOW', color='#27AE60', fontsize=9, fontweight='bold')
    ax.text( 0.75, -0.25, 'HIGH', color='#E74C3C', fontsize=9, fontweight='bold')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.5, 1.1)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Key Risk Factors
    st.markdown("### 🧠 Key Risk Factors")
    risk_factors = []
    if overtime == "Yes":            risk_factors.append(("OverTime = Yes", "High", "#E74C3C"))
    if job_satisfaction <= 2:        risk_factors.append(("Low Job Satisfaction", "High", "#E74C3C"))
    if env_satisfaction <= 2:        risk_factors.append(("Low Environment Satisfaction", "High", "#E74C3C"))
    if business_travel == "Travel_Frequently": risk_factors.append(("Frequent Business Travel", "Medium", "#F39C12"))
    if marital_status == "Single":   risk_factors.append(("Single Marital Status", "Medium", "#F39C12"))
    if years_since_promo > 3:        risk_factors.append(("Overdue Promotion (>3 yrs)", "Medium", "#F39C12"))
    if monthly_income < 3500:        risk_factors.append(("Low Monthly Income", "High", "#E74C3C"))
    if distance_from_home > 20:      risk_factors.append(("Far from Home", "Low", "#27AE60"))
    if work_life_balance == 1:       risk_factors.append(("Poor Work-Life Balance", "High", "#E74C3C"))
    if years_at_company < 2:         risk_factors.append(("New Employee (<2 yrs)", "Medium", "#F39C12"))
    if not risk_factors:             risk_factors.append(("No major risk factors identified", "Low", "#27AE60"))

    for factor, level, color in risk_factors:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:8px 12px; margin:4px 0; border-radius:6px; background:#f8f9fa;
                    border-left:4px solid {color};">
            <span style="color:#2C3E50;">{factor}</span>
            <span style="color:{color}; font-weight:bold; font-size:0.85rem;">{level} Risk</span>
        </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("### 📋 Employee Summary")
    summary_data = {
        "Age": age, "Department": department, "Job Role": job_role,
        "Job Level": job_level, "Monthly Income": f"${monthly_income:,}",
        "Years at Company": years_at_company,
        "Job Satisfaction": f"{job_satisfaction}/4",
        "OverTime": overtime, "Business Travel": business_travel,
        "Marital Status": marital_status,
        "Satisfaction Index": f"{(job_satisfaction+env_satisfaction+relationship_sat+work_life_balance)/4:.2f}/4",
    }
    for k, v in summary_data.items():
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"**{k}**")
        c2.markdown(str(v))
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    perf_data = {
        "Model Used": best_model_name,
        "Accuracy": "65.0%",
        "F1 Score": "0.605",
        "ROC-AUC": "0.699",
        "Cross-Val AUC": "0.724 ± 0.031",
    }
    for k, v in perf_data.items():
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"**{k}**")
        c2.markdown(str(v))

    st.markdown("---")
    st.markdown("### 💡 HR Recommendations")
    if pred == 1:
        st.warning("**Immediate Actions Suggested:**")
        recs = []
        if overtime == "Yes":        recs.append("Consider reducing overtime load")
        if job_satisfaction <= 2:    recs.append("Schedule satisfaction survey / 1:1 meeting")
        if years_since_promo > 3:    recs.append("Review promotion eligibility")
        if monthly_income < 3500:    recs.append("Review compensation package")
        if work_life_balance <= 2:   recs.append("Offer flexible work arrangements")
        if not recs:                 recs = ["Proactive engagement recommended", "Enroll in retention program"]
        for r in recs:
            st.markdown(f"• {r}")
    else:
        st.success("**Employee appears engaged. Continue:**")
        st.markdown("• Regular performance reviews\n• Learning & development opportunities\n• Recognition programs")

# ─── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    Employee Attrition Prediction System | AIML Summer Internship 2026 | IIHMF, MNNIT Allahabad
</div>
""", unsafe_allow_html=True)
