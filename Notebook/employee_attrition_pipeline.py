"""
====================================================================
EMPLOYEE ATTRITION PREDICTION SYSTEM
AIML Summer Internship 2026 - IIHMF, MNNIT Allahabad
Capstone Project 4

Dataset: IBM HR Analytics Employee Attrition & Performance
Source  : Kaggle – WA_Fn-UseC_-HR-Employee-Attrition.csv
Records : 1,470 | Features: 35
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
# Resolve paths relative to this script so the pipeline works on any machine.
# Override BASE by setting the ATTRITION_BASE environment variable.
BASE  = os.environ.get('ATTRITION_BASE', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DATA  = os.path.join(BASE, 'Dataset', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')   # Real Kaggle dataset
MODEL = os.path.join(BASE, 'Model')
PLOTS = os.path.join(BASE, 'Documentation', 'plots')
os.makedirs(MODEL, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

PALETTE = {'Yes': '#E74C3C', 'No': '#2ECC71'}
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

print("=" * 60)
print("  EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("  AIML Summer Internship 2026")
print("  Dataset: IBM HR Analytics (Kaggle – Real Dataset)")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# PHASE 1 & 2 ─ PROBLEM UNDERSTANDING & DATA COLLECTION
# ════════════════════════════════════════════════════════════════
print("\n[Phase 1-2] Loading Kaggle IBM HR Dataset...")
df = pd.read_csv(DATA)
# Strip BOM if present
df.columns = df.columns.str.lstrip('\ufeff').str.strip()

print(f"  Shape         : {df.shape}")
print(f"  Columns       : {df.shape[1]}")
print(f"  Attrition(Yes): {(df['Attrition']=='Yes').sum()} "
      f"({(df['Attrition']=='Yes').mean():.1%})")
print(f"  Attrition(No) : {(df['Attrition']=='No').sum()} "
      f"({(df['Attrition']=='No').mean():.1%})")

# ════════════════════════════════════════════════════════════════
# PHASE 3 ─ DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════
print("\n[Phase 3] Data Preprocessing...")

print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"  Duplicates     : {df.duplicated().sum()}")

# Drop constant / ID columns
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df.drop(columns=drop_cols, inplace=True)
print(f"  Dropped constant/ID cols: {drop_cols}")

# Encode binary target
df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)

# Outlier treatment using IQR capping
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != 'Attrition_Binary']
outlier_count = 0
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    outlier_count += n_out
    df[col] = df[col].clip(lo, hi)
print(f"  Outliers capped (IQR): {outlier_count} values")

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Attrition']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    le_dict[col] = le
print(f"  Encoded categorical cols: {cat_cols}")

# ════════════════════════════════════════════════════════════════
# PHASE 4 ─ EDA
# ════════════════════════════════════════════════════════════════
print("\n[Phase 4] Exploratory Data Analysis...")

# 4.1 Univariate histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Univariate Analysis – Key Numerical Features', fontsize=14, fontweight='bold')
univ_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
             'TotalWorkingYears', 'PercentSalaryHike']
for ax, col in zip(axes.flatten(), univ_cols):
    ax.hist(df[col], bins=25, color='#3498DB', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '01_univariate_histograms.png'), bbox_inches='tight')
plt.close()
print("  Saved: 01_univariate_histograms.png")

# 4.2 Attrition class distribution
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#2ECC71', '#E74C3C']
counts = df['Attrition'].value_counts()
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.4)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f'{bar.get_height()}', ha='center', va='bottom', fontweight='bold')
ax.set_title('Attrition Distribution', fontsize=13, fontweight='bold')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '02_attrition_distribution.png'), bbox_inches='tight')
plt.close()
print("  Saved: 02_attrition_distribution.png")

# 4.3 Bivariate analysis
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Bivariate Analysis – Attrition vs Key Features', fontsize=14, fontweight='bold')
biv_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
            'EnvironmentSatisfaction', 'WorkLifeBalance']
for ax, col in zip(axes.flatten(), biv_cols):
    for label, color in PALETTE.items():
        subset = df[df['Attrition'] == label][col]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend(title='Attrition')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '03_bivariate_analysis.png'), bbox_inches='tight')
plt.close()
print("  Saved: 03_bivariate_analysis.png")

# 4.4 Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Boxplot Analysis by Attrition', fontsize=14, fontweight='bold')
for ax, col in zip(axes, ['MonthlyIncome', 'Age', 'DistanceFromHome']):
    data_no  = df[df['Attrition'] == 'No'][col]
    data_yes = df[df['Attrition'] == 'Yes'][col]
    bp = ax.boxplot([data_no, data_yes], labels=['No', 'Yes'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ECC71')
    bp['boxes'][1].set_facecolor('#E74C3C')
    ax.set_title(col, fontsize=11)
    ax.set_xlabel('Attrition')
    ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '04_boxplots.png'), bbox_inches='tight')
plt.close()
print("  Saved: 04_boxplots.png")

# 4.5 Correlation heatmap
num_df = df[num_cols + ['Attrition_Binary']].copy()
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
            linewidths=0.3, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '05_correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print("  Saved: 05_correlation_heatmap.png")

# 4.6 Attrition by categorical features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Attrition Rate by Categorical Features', fontsize=14, fontweight='bold')
for ax, col in zip(axes, ['Department', 'OverTime', 'MaritalStatus']):
    rate = df.groupby(col)['Attrition_Binary'].mean().sort_values()
    ax.barh(rate.index, rate.values * 100, color='#E74C3C', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Attrition Rate (%)')
    ax.set_title(f'By {col}', fontsize=11)
    for i, v in enumerate(rate.values):
        ax.text(v * 100 + 0.5, i, f'{v:.1%}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '06_categorical_attrition.png'), bbox_inches='tight')
plt.close()
print("  Saved: 06_categorical_attrition.png")

# ════════════════════════════════════════════════════════════════
# PHASE 5 ─ FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
print("\n[Phase 5] Feature Engineering...")

df['IncomeToExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
df['SatisfactionIndex']  = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] +
                             df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
df['TenureRatio']  = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['OverdueProm']  = (df['YearsSinceLastPromotion'] > 3).astype(int)
df['IsJunior']     = (df['JobLevel'] <= 2).astype(int)
df['LowIncome']    = (df['MonthlyIncome'] < df['MonthlyIncome'].quantile(0.25)).astype(int)
print("  Created: IncomeToExperience, SatisfactionIndex, TenureRatio, OverdueProm, IsJunior, LowIncome")

enc_features = [c + '_enc' for c in cat_cols]
new_features = ['IncomeToExperience', 'SatisfactionIndex', 'TenureRatio',
                'OverdueProm', 'IsJunior', 'LowIncome']
feature_cols = num_cols + enc_features + new_features
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df['Attrition_Binary'].copy()
print(f"  Total features : {X.shape[1]}")

# Feature importance via RF for selection
tmp_rf = RandomForestClassifier(n_estimators=50, random_state=42)
tmp_rf.fit(X, y)
importances = pd.Series(tmp_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(20).index.tolist()
X = X[top_features]
print(f"  Selected top 20 features via RF importance")

# Feature importance plot
fig, ax = plt.subplots(figsize=(10, 7))
importances.head(20).sort_values().plot(kind='barh', ax=ax, color='#3498DB', edgecolor='white')
ax.set_title('Top 20 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '07_feature_importance.png'), bbox_inches='tight')
plt.close()
print("  Saved: 07_feature_importance.png")

# ════════════════════════════════════════════════════════════════
# PHASE 6 ─ MODEL BUILDING
# ════════════════════════════════════════════════════════════════
print("\n[Phase 6] Model Building...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {0: cw[0], 1: cw[1]}
print(f"  Class weights: {cw_dict}")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced', C=0.5),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42),
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_prob),
    }
    print(f"  {name}: Acc={results[name]['accuracy']:.4f} | "
          f"F1={results[name]['f1']:.4f} | ROC-AUC={results[name]['roc_auc']:.4f}")

# ════════════════════════════════════════════════════════════════
# PHASE 7 ─ MODEL EVALUATION
# ════════════════════════════════════════════════════════════════
print("\n[Phase 7] Model Evaluation...")

metrics_df = pd.DataFrame({
    name: {
        'Accuracy':  r['accuracy'],
        'Precision': r['precision'],
        'Recall':    r['recall'],
        'F1 Score':  r['f1'],
        'ROC-AUC':   r['roc_auc'],
    } for name, r in results.items()
}).T.round(4)
print("\n  Model Comparison:\n")
print(metrics_df.to_string())
metrics_df.to_csv(os.path.join(BASE, 'Documentation', 'model_metrics.csv'))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices – All Models', fontsize=13, fontweight='bold')
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '08_confusion_matrices.png'), bbox_inches='tight')
plt.close()
print("  Saved: 08_confusion_matrices.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
colors_line = ['#3498DB', '#E74C3C', '#2ECC71']
for (name, r), color in zip(results.items(), colors_line):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves – All Models', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '09_roc_curves.png'), bbox_inches='tight')
plt.close()
print("  Saved: 09_roc_curves.png")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
x = np.arange(len(metric_names))
width = 0.25
colors_bar = ['#3498DB', '#E74C3C', '#2ECC71']
for i, (name, r) in enumerate(results.items()):
    vals = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['roc_auc']]
    ax.bar(x + i * width, vals, width, label=name, color=colors_bar[i],
           alpha=0.85, edgecolor='white')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.1)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_ylabel('Score')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, '10_model_comparison.png'), bbox_inches='tight')
plt.close()
print("  Saved: 10_model_comparison.png")

# Cross-validation
print("\n  Cross-Validation (5-Fold Stratified):")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, r in results.items():
    mod = r['model']
    if name == 'Logistic Regression':
        cv_scores = cross_val_score(mod, X_train_s, y_train, cv=skf, scoring='roc_auc')
    else:
        cv_scores = cross_val_score(mod, X_train, y_train, cv=skf, scoring='roc_auc')
    cv_results[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
    print(f"    {name}: Mean AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Best model selection
best_name = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_name]['model']
print(f"\n  Best Model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
print("\n  Classification Report:")
print(classification_report(y_test, results[best_name]['y_pred'],
                            target_names=['Stay', 'Leave']))

# ════════════════════════════════════════════════════════════════
# PHASE 8 ─ SAVE MODEL & ARTIFACTS
# ════════════════════════════════════════════════════════════════
print("\n[Phase 8] Saving Model Artifacts...")

joblib.dump(best_model,   os.path.join(MODEL, 'best_model.pkl'))
joblib.dump(scaler,       os.path.join(MODEL, 'scaler.pkl'))
joblib.dump(top_features, os.path.join(MODEL, 'feature_names.pkl'))
joblib.dump(le_dict,      os.path.join(MODEL, 'label_encoders.pkl'))

feature_info = {
    'cat_cols':        cat_cols,
    'top_features':    top_features,
    'best_model_name': best_name,
}
joblib.dump(feature_info, os.path.join(MODEL, 'feature_info.pkl'))

print(f"  Saved: best_model.pkl ({best_name})")
print(f"  Saved: scaler.pkl, feature_names.pkl, label_encoders.pkl, feature_info.pkl")

# Print final metrics summary for report
print("\n" + "=" * 60)
print("  FINAL METRICS SUMMARY")
print("=" * 60)
for name, r in results.items():
    print(f"  {name}:")
    print(f"    Accuracy  = {r['accuracy']:.4f}")
    print(f"    Precision = {r['precision']:.4f}")
    print(f"    Recall    = {r['recall']:.4f}")
    print(f"    F1 Score  = {r['f1']:.4f}")
    print(f"    ROC-AUC   = {r['roc_auc']:.4f}")
    print(f"    CV AUC    = {cv_results[name]['mean']:.4f} ± {cv_results[name]['std']:.4f}")
    print()

print("=" * 60)
print("  ALL PHASES COMPLETE")
print("=" * 60)
