"""
Employee Attrition Prediction System - Flask Web Application
AIML Summer Internship 2026, IIHMF, MNNIT Allahabad
Capstone Project 4
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import sys

app = Flask(__name__)

# ─── Load Model Artifacts ─────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE, 'Model')

model      = joblib.load(f'{MODEL_DIR}/best_model.pkl')
scaler     = joblib.load(f'{MODEL_DIR}/scaler.pkl')
features   = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
le_dict    = joblib.load(f'{MODEL_DIR}/label_encoders.pkl')
feat_info  = joblib.load(f'{MODEL_DIR}/feature_info.pkl')

# ─── Helper ───────────────────────────────────────────────────────
def encode_cat(col, val):
    le = le_dict.get(col)
    if le and val in le.classes_:
        return int(le.transform([val])[0])
    return 0

def get_risk_factors(data):
    risks = []
    if data['OverTime'] == 'Yes':
        risks.append(('OverTime = Yes', 'High', '#E74C3C'))
    if int(data['JobSatisfaction']) <= 2:
        risks.append(('Low Job Satisfaction', 'High', '#E74C3C'))
    if int(data['EnvironmentSatisfaction']) <= 2:
        risks.append(('Low Environment Satisfaction', 'High', '#E74C3C'))
    if int(data['WorkLifeBalance']) == 1:
        risks.append(('Poor Work-Life Balance', 'High', '#E74C3C'))
    if float(data['MonthlyIncome']) < 3500:
        risks.append(('Low Monthly Income', 'High', '#E74C3C'))
    if data['BusinessTravel'] == 'Travel_Frequently':
        risks.append(('Frequent Business Travel', 'Medium', '#F39C12'))
    if data['MaritalStatus'] == 'Single':
        risks.append(('Single Marital Status', 'Medium', '#F39C12'))
    if int(data['YearsSinceLastPromotion']) > 3:
        risks.append(('Overdue Promotion (>3 yrs)', 'Medium', '#F39C12'))
    if int(data['YearsAtCompany']) < 2:
        risks.append(('New Employee (<2 yrs)', 'Medium', '#F39C12'))
    if int(data['DistanceFromHome']) > 20:
        risks.append(('Far from Home', 'Low', '#27AE60'))
    if not risks:
        risks.append(('No major risk factors detected', 'Low', '#27AE60'))
    return risks

def get_recommendations(pred, data):
    recs = []
    if pred == 1:
        if data['OverTime'] == 'Yes':
            recs.append('Reduce overtime workload or provide compensatory benefits')
        if int(data['JobSatisfaction']) <= 2:
            recs.append('Schedule a 1-on-1 satisfaction review with the employee')
        if int(data['YearsSinceLastPromotion']) > 3:
            recs.append('Review promotion eligibility immediately')
        if float(data['MonthlyIncome']) < 3500:
            recs.append('Review and revise compensation package')
        if int(data['WorkLifeBalance']) <= 2:
            recs.append('Offer flexible work arrangements or remote work option')
        if not recs:
            recs = ['Proactive engagement recommended', 'Enroll in employee retention program',
                    'Conduct stay interview']
    else:
        recs = [
            'Continue regular performance reviews',
            'Offer learning & development opportunities',
            'Recognize achievements through rewards program',
            'Maintain current engagement level',
        ]
    return recs

# ─── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.form

        # Numerical inputs
        age                  = int(d['age'])
        monthly_income       = float(d['monthly_income'])
        daily_rate           = int(d.get('daily_rate', 800))
        distance_from_home   = int(d['distance_from_home'])
        education            = int(d['education'])
        env_satisfaction     = int(d['env_satisfaction'])
        job_involvement      = int(d['job_involvement'])
        job_level            = int(d['job_level'])
        job_satisfaction     = int(d['job_satisfaction'])
        num_companies_worked = int(d['num_companies_worked'])
        percent_salary_hike  = int(d['percent_salary_hike'])
        performance_rating   = int(d.get('performance_rating', 3))
        relationship_sat     = int(d['relationship_sat'])
        stock_option_level   = int(d['stock_option_level'])
        total_working_years  = int(d['total_working_years'])
        training_times       = int(d['training_times'])
        work_life_balance    = int(d['work_life_balance'])
        years_at_company     = int(d['years_at_company'])
        years_in_role        = int(d['years_in_role'])
        years_since_promo    = int(d['years_since_promo'])
        years_with_manager   = int(d['years_with_manager'])

        # Categorical inputs
        business_travel  = d['business_travel']
        department       = d['department']
        education_field  = d['education_field']
        gender           = d['gender']
        job_role         = d['job_role']
        marital_status   = d['marital_status']
        overtime         = d['overtime']

        # Engineered features
        income_to_exp    = monthly_income / (total_working_years + 1)
        satisfaction_idx = (job_satisfaction + env_satisfaction + relationship_sat + work_life_balance) / 4
        tenure_ratio     = years_in_role / (years_at_company + 1)
        overdue_prom     = int(years_since_promo > 3)
        is_junior        = int(job_level <= 2)
        low_income       = int(monthly_income < 3500)

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
            'BusinessTravel_enc': encode_cat('BusinessTravel', business_travel),
            'Department_enc': encode_cat('Department', department),
            'EducationField_enc': encode_cat('EducationField', education_field),
            'Gender_enc': encode_cat('Gender', gender),
            'JobRole_enc': encode_cat('JobRole', job_role),
            'MaritalStatus_enc': encode_cat('MaritalStatus', marital_status),
            'OverTime_enc': encode_cat('OverTime', overtime),
            'IncomeToExperience': income_to_exp,
            'SatisfactionIndex': satisfaction_idx,
            'TenureRatio': tenure_ratio,
            'OverdueProm': overdue_prom,
            'IsJunior': is_junior,
            'LowIncome': low_income,
        }

        import pandas as pd
        input_df = pd.DataFrame([input_dict])
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df  = input_df[features]
        input_scaled = scaler.transform(input_df)

        prob = float(model.predict_proba(input_scaled)[0][1])
        pred = int(prob >= 0.5)

        risk_level = 'High' if prob >= 0.65 else ('Medium' if prob >= 0.40 else 'Low')
        risk_color = '#E74C3C' if risk_level == 'High' else ('#F39C12' if risk_level == 'Medium' else '#27AE60')

        risk_factors    = get_risk_factors(dict(d))
        recommendations = get_recommendations(pred, dict(d))

        employee = {
            'age': age, 'department': department, 'job_role': job_role,
            'job_level': job_level, 'monthly_income': f"${monthly_income:,.0f}",
            'years_at_company': years_at_company, 'overtime': overtime,
            'job_satisfaction': f"{job_satisfaction}/4",
            'satisfaction_index': f"{satisfaction_idx:.2f}/4",
            'marital_status': marital_status, 'business_travel': business_travel,
        }

        return render_template('result.html',
            pred=pred,
            prob=round(prob * 100, 1),
            risk_level=risk_level,
            risk_color=risk_color,
            risk_factors=risk_factors,
            recommendations=recommendations,
            employee=employee,
            model_name=feat_info['best_model_name'],
        )

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
