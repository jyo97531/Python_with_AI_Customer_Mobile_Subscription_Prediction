# ============================================================
# INDIVIDUAL ML PROJECT: Mobile Subscription Churn Prediction
# Task Type: Classification
# Target: Will customer continue subscription? (1=Yes, 0=No)
# Dataset: Telco Customer Churn (Kaggle)
# Models: Decision Tree + Logistic Regression + Random Forest
# ============================================================

# ── STEP 1: Install & Import Libraries ───────────────────


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Step 2: Load Data (Read the CSV file using pandas) ───────────────────────────

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("=" * 55)
print("  MOBILE SUBSCRIPTION CHURN PREDICTION PROJECT")
print("=" * 55)
print(f"\n📦 Dataset loaded: {df.shape[0]} customers, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ── STEP 3: Explore the Data ──────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: DATA EXPLORATION")
print("=" * 55)

print("\n📊 Churn Distribution:")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"\n  ✅ Continuing (No Churn):  {churn_counts['No']} customers ({churn_counts['No']/len(df):.1%})")
print(f"  ❌ Left (Churned):          {churn_counts['Yes']} customers ({churn_counts['Yes']/len(df):.1%})")

print("\n📋 Data Types & Missing Values:")
print(df.isnull().sum())

# ── STEP 4: Clean & Preprocess Data ──────────────────────
print("\n" + "=" * 55)
print("STEP 4: DATA CLEANING")
print("=" * 55)

# Drop customerID (not useful for prediction)
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric (sometimes it has spaces in real dataset)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# ✅ Convert target: Churn Yes/No → 1/0
# 1 = Customer CONTINUES (No churn)
# 0 = Customer LEAVES (Churned)
df['Churn'] = df['Churn'].map({'No': 1, 'Yes': 0})

print("Target column converted:")
print("  'No'  (continues)  → 1")
print("  'Yes' (leaves)     → 0")

# Encode all categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print(f"\n✅ Encoded {len(categorical_cols)} categorical columns")
print("Cleaned dataset shape:", df.shape)
print("\nSample cleaned data:")
print(df.head(3))

# ── STEP 5: Feature Selection ─────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: FEATURE SELECTION")
print("=" * 55)

X = df.drop('Churn', axis=1)   # All columns except target
y = df['Churn']                 # Target: 1=continues, 0=leaves

print(f"Features (X): {X.columns.tolist()}")
print(f"Target  (y): Churn (1=continues, 0=leaves)")

# ── STEP 6: Scale Features ────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ── STEP 7: Train/Test Split ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing  samples: {len(X_test)}")

# ── STEP 8: Train 3 Models ────────────────────────────────
print("\n" + "=" * 55)
print("STEP 8: TRAINING MODELS")
print("=" * 55)

# Model 1: Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
print("✅ Decision Tree trained")

# Model 2: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
print("✅ Logistic Regression trained")

# Model 3: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("✅ Random Forest trained")

# ── STEP 9: Predictions & Evaluation ─────────────────────
print("\n" + "=" * 55)
print("STEP 9: MODEL EVALUATION")
print("=" * 55)

models = {
    'Decision Tree':      (dt, dt.predict(X_test)),
    'Logistic Regression':(lr, lr.predict(X_test)),
    'Random Forest':      (rf, rf.predict(X_test))
}

results = {}
for name, (model, pred) in models.items():
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"\n📊 {name}")
    print(f"   Accuracy: {acc:.2%}")
    print(classification_report(y_test, pred,
          target_names=['Left (0)', 'Continues (1)']))

best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model_name} ({results[best_model_name]:.2%})")

# ── STEP 10: Visualizations ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Mobile Subscription Churn Prediction\nModel Analysis',
             fontsize=15, fontweight='bold', y=1.01)

# Plot 1: Churn Distribution
ax1 = axes[0, 0]
labels = ['Continues\n(1)', 'Leaves\n(0)']
sizes  = [y.sum(), (y == 0).sum()]
colors = ['#2ecc71', '#e74c3c']
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11})
ax1.set_title('Customer Distribution', fontweight='bold')

# Plot 2: Accuracy Comparison
ax2 = axes[0, 1]
names = list(results.keys())
accs  = [v * 100 for v in results.values()]
bars  = ax2.bar(names, accs, color=['#3498db', '#9b59b6', '#e67e22'], width=0.5)
ax2.set_ylim(0, 100)
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Model Accuracy Comparison', fontweight='bold')
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', fontweight='bold')
ax2.tick_params(axis='x', rotation=15)

# Plot 3: Confusion Matrix - Best Model
ax3 = axes[1, 0]
best_pred = models[best_model_name][1]
cm = confusion_matrix(y_test, best_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
#             xticklabels=['Leaves (0)', 'Continues (1)'],
#             yticklabels=['Leaves (0)', 'Continues (1)'])
ax3.set_title(f'Confusion Matrix\n({best_model_name})', fontweight='bold')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# Plot 4: Feature Importance (Random Forest)
ax4 = axes[1, 1]
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)
top10.sort_values().plot(kind='barh', ax=ax4, color='#1abc9c')
ax4.set_title('Top 10 Important Features\n(Random Forest)', fontweight='bold')
ax4.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('churn_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Charts saved as 'churn_prediction_results.png'")

# ── STEP 11: Predict New Customers ───────────────────────
print("\n" + "=" * 55)
print("STEP 11: PREDICT NEW CUSTOMERS")
print("=" * 55)

# Best model for prediction
best = models[best_model_name][0]

# Create sample customers using same feature columns
sample_customers = pd.DataFrame([
    # tenure, MonthlyCharges, Contract=2(Two year), SeniorCitizen=0
    [0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 2, 0, 0, 45.0, 2700.0],   # Loyal customer
    [1, 1, 1, 1,  2, 1, 1, 2, 1, 1, 0, 1, 1, 95.0,  190.0],   # New + high charge
], columns=X.columns)

sample_scaled = scaler.transform(sample_customers)
predictions   = best.predict(sample_scaled)
probabilities = best.predict_proba(sample_scaled)

print(f"\nUsing Best Model: {best_model_name}")
print("-" * 45)
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "✅ CONTINUES (1)" if pred == 1 else "❌ WILL LEAVE (0)"
    conf   = max(prob) * 100
    print(f"\nCustomer {i+1}: {status}")
    print(f"  Confidence: {conf:.1f}%")
    print(f"  Prob(Leaves=0): {prob[0]:.2%}  |  Prob(Continues=1): {prob[1]:.2%}")

print("\n" + "=" * 55)
print("  PROJECT COMPLETE ✅")
print("  Prediction: 1 = Customer Continues Subscription")
print("              0 = Customer Will Leave")
print("=" * 55)

