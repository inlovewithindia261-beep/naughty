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
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load Model Artifacts ─────────────────────────────────────────
try:
    # Get the correct path to the Model directory
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # If deployment root is Flask_App, Model is at ../Model
    MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'Model'))
    # Normalize the path to resolve .. references
    MODEL_DIR = os.path.normpath(MODEL_DIR)
    print("CURRENT_DIR=", CURRENT_DIR)
    print("MODEL_DIR=", MODEL_DIR)
    logger.info(f"Current directory: {CURRENT_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Model directory exists: {os.path.exists(MODEL_DIR)}")

    # Load model files with error handling
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    le_dict_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    feat_info_path = os.path.join(MODEL_DIR, 'feature_info.pkl')

    logger.info(f"Model file exists: {os.path.exists(model_path)}")
    logger.info(f"Scaler file exists: {os.path.exists(scaler_path)}")
    logger.info(f"Features file exists: {os.path.exists(features_path)}")
    logger.info(f"Label encoders file exists: {os.path.exists(le_dict_path)}")
    logger.info(f"Feature info file exists: {os.path.exists(feat_info_path)}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    le_dict = joblib.load(le_dict_path)
    feat_info = joblib.load(feat_info_path)

    logger.info("✓ All model artifacts loaded successfully")

except Exception as e:
    logger.error(f"✗ Error loading model artifacts: {str(e)}")
    model = None
    scaler = None
    features = None
    le_dict = None
    feat_info = None

# ─── Helper ───────────────────────────────────────────────────────
def encode_cat(col, val):
    """Safely encode categorical variables"""
    try:
        if not le_dict:
            return 0
        le = le_dict.get(col)
        if le and val in le.classes_:
            return int(le.transform([val])[0])
        return 0
    except Exception as e:
        logger.warning(f"Error encoding {col}={val}: {str(e)}")
        return 0

def get_risk_factors(data):
    """Identify risk factors for employee attrition"""
    risks = []
    try:
        if data.get('overtime') == 'Yes':
            risks.append(('OverTime = Yes', 'High', '#E74C3C'))
        if int(data.get('job_satisfaction', 0)) <= 2:
            risks.append(('Low Job Satisfaction', 'High', '#E74C3C'))
        if int(data.get('env_satisfaction', 0)) <= 2:
            risks.append(('Low Environment Satisfaction', 'High', '#E74C3C'))
        if int(data.get('work_life_balance', 0)) == 1:
            risks.append(('Poor Work-Life Balance', 'High', '#E74C3C'))
        if float(data.get('monthly_income', 0)) < 3500:
            risks.append(('Low Monthly Income', 'High', '#E74C3C'))
        if data.get('business_travel') == 'Travel_Frequently':
            risks.append(('Frequent Business Travel', 'Medium', '#F39C12'))
        if data.get('marital_status') == 'Single':
            risks.append(('Single Marital Status', 'Medium', '#F39C12'))
        if int(data.get('years_since_promo', 0)) > 3:
            risks.append(('Overdue Promotion (>3 yrs)', 'Medium', '#F39C12'))
        if int(data.get('years_at_company', 0)) < 2:
            risks.append(('New Employee (<2 yrs)', 'Medium', '#F39C12'))
        if int(data.get('distance_from_home', 0)) > 20:
            risks.append(('Far from Home', 'Low', '#27AE60'))
        if not risks:
            risks.append(('No major risk factors detected', 'Low', '#27AE60'))
    except Exception as e:
        logger.warning(f"Error calculating risk factors: {str(e)}")
        risks.append(('Unable to calculate risk factors', 'Low', '#27AE60'))
    return risks

def get_recommendations(pred, data):
    """Generate HR recommendations based on prediction"""
    recs = []
    try:
        if pred == 1:
            if data.get('overtime') == 'Yes':
                recs.append('Reduce overtime workload or provide compensatory benefits')
            if int(data.get('job_satisfaction', 0)) <= 2:
                recs.append('Schedule a 1-on-1 satisfaction review with the employee')
            if int(data.get('years_since_promo', 0)) > 3:
                recs.append('Review promotion eligibility immediately')
            if float(data.get('monthly_income', 0)) < 3500:
                recs.append('Review and revise compensation package')
            if int(data.get('work_life_balance', 0)) <= 2:
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
    except Exception as e:
        logger.warning(f"Error generating recommendations: {str(e)}")
        recs = ['Unable to generate recommendations']
    return recs

# ─── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on form data"""
    try:
        # Check if model is loaded
        if not all([model, scaler, features, le_dict, feat_info]):
            return render_template('index.html', error="Model not loaded. Please try again later.")

        d = request.form

        # Numerical inputs with default values
        age = int(d.get('age', 30))
        monthly_income = float(d.get('monthly_income', 5000))
        daily_rate = int(d.get('daily_rate', 800))
        distance_from_home = int(d.get('distance_from_home', 10))
        education = int(d.get('education', 3))
        env_satisfaction = int(d.get('env_satisfaction', 3))
        job_involvement = int(d.get('job_involvement', 3))
        job_level = int(d.get('job_level', 2))
        job_satisfaction = int(d.get('job_satisfaction', 3))
        num_companies_worked = int(d.get('num_companies_worked', 2))
        percent_salary_hike = int(d.get('percent_salary_hike', 11))
        performance_rating = int(d.get('performance_rating', 3))
        relationship_sat = int(d.get('relationship_sat', 3))
        stock_option_level = int(d.get('stock_option_level', 1))
        total_working_years = int(d.get('total_working_years', 10))
        training_times = int(d.get('training_times', 3))
        work_life_balance = int(d.get('work_life_balance', 3))
        years_at_company = int(d.get('years_at_company', 5))
        years_in_role = int(d.get('years_in_role', 3))
        years_since_promo = int(d.get('years_since_promo', 2))
        years_with_manager = int(d.get('years_with_manager', 3))

        # Categorical inputs
        business_travel = d.get('business_travel', 'Travel_Rarely')
        department = d.get('department', 'Sales')
        education_field = d.get('education_field', 'Life Sciences')
        gender = d.get('gender', 'Male')
        job_role = d.get('job_role', 'Sales Executive')
        marital_status = d.get('marital_status', 'Single')
        overtime = d.get('overtime', 'No')

        # Engineered features
        income_to_exp = monthly_income / (total_working_years + 1)
        satisfaction_idx = (job_satisfaction + env_satisfaction + relationship_sat + work_life_balance) / 4
        tenure_ratio = years_in_role / (years_at_company + 1)
        overdue_prom = int(years_since_promo > 3)
        is_junior = int(job_level <= 2)
        low_income = int(monthly_income < 3500)

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

        # Ensure all required features are present
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)

        prob = float(model.predict_proba(input_scaled)[0][1])
        pred = int(prob >= 0.5)

        risk_level = 'High' if prob >= 0.65 else ('Medium' if prob >= 0.40 else 'Low')
        risk_color = '#E74C3C' if risk_level == 'High' else ('#F39C12' if risk_level == 'Medium' else '#27AE60')

        risk_factors = get_risk_factors(d)
        recommendations = get_recommendations(pred, d)

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
            model_name=feat_info.get('best_model_name', 'ML Model'),
        )

    except ValueError as ve:
        logger.error(f"Value error in prediction: {str(ve)}")
        return render_template('index.html', error=f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('index.html', error=f"Prediction error: {str(e)}")


# ─── Health Check (Debug) ──────────────────────────────────────────
@app.route('/health')
def health():
    """Detailed health check endpoint for debugging deployment"""
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'Model'))

    # List files in Model dir if it exists
    if os.path.exists(MODEL_DIR):
        try:
            model_files = os.listdir(MODEL_DIR)
        except Exception as e:
            model_files = f"Error listing: {str(e)}"
    else:
        model_files = "DIRECTORY NOT FOUND"

    # List files in repo root (one level up from Flask_App)
    repo_root = os.path.normpath(os.path.join(CURRENT_DIR, '..'))
    try:
        root_contents = os.listdir(repo_root)
    except Exception as e:
        root_contents = f"Error: {str(e)}"

    return {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_loaded': features is not None,
        'current_dir': CURRENT_DIR,
        'model_dir': MODEL_DIR,
        'model_dir_exists': os.path.exists(MODEL_DIR),
        'model_files_found': model_files,
        'repo_root': repo_root,
        'repo_root_contents': root_contents,
    }, 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
