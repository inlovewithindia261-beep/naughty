# Employee Attrition Prediction System
### AIML Summer Internship 2026 | IIHMF, MNNIT Allahabad | Capstone Project 4

---

## 📌 Project Overview
This project builds an end-to-end Machine Learning system to predict whether an employee
is likely to leave an organization, using classification models trained on HR data.

## 📁 Folder Structure
```
EmployeeAttritionPrediction/
├── Dataset/
│   ├── generate_dataset.py          # Script to generate synthetic dataset
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv (IBM HR Analytics – Kaggle)       # Dataset (1470 records, 35 features)
├── Notebook/
│   └── employee_attrition_pipeline.py  # Full ML pipeline (Phases 1-8)
├── Model/
│   ├── best_model.pkl               # Trained Logistic Regression model
│   ├── scaler.pkl                   # StandardScaler
│   ├── feature_names.pkl            # Top 20 selected features
│   ├── label_encoders.pkl           # Label encoders for categorical cols
│   └── feature_info.pkl             # Feature metadata
├── Streamlit_App/
│   └── app.py                       # Streamlit web application
├── Documentation/
│   ├── plots/                       # 10 EDA & evaluation plots
│   ├── model_metrics.csv            # Model performance table
│   └── Project_Report_EmployeeAttritionPrediction.pdf
└── README.md
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit reportlab
```

### 2. Generate Dataset
```bash
python Dataset/generate_dataset.py
```

### 3. Run the ML Pipeline
```bash
python Notebook/employee_attrition_pipeline.py
```

### 4. Launch Streamlit App
```bash
cd Streamlit_App
streamlit run app.py
```

## 🤖 Models Trained
| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression ⭐ | 0.650 | 0.605 | 0.699 |
| Random Forest | 0.585 | 0.516 | 0.672 |
| Gradient Boosting | 0.605 | 0.528 | 0.689 |

## 📊 ML Phases Covered
1. Problem Understanding
2. Data Collection
3. Data Preprocessing
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Model Building
7. Model Evaluation
8. Model Deployment (Streamlit)

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **ML Library**: Scikit-Learn
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit
- **Report**: ReportLab

## 📝 Submission Format
`EmployeeAttritionPrediction_TeamLeaderName.zip`
