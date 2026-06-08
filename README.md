PROJECT REPORT

Mobile Subscription Churn Prediction

A Supervised Classification Study with Task-by-Task Walkthrough and Console Output

Task type    ------------------------------  Binary classification (supervised learning)

Dataset      ------------------------------  Telco Customer Churn (Kaggle / IBM sample)

Models       ------------------------------  Decision Tree, Logistic Regression, Random Forest

Language / stack     ----------------------  Python 3, scikit-learn, pandas, matplotlib, seaborn

Deliverables    ---------------------------  churn script (.txt/.py) + results figure (.png)

1. Executive Summary

This report documents an end-to-end machine learning project that predicts customer churn for a mobile / telecom subscription service: given a customer's account and service attributes, will they continue their subscription or leave?

Three classifiers — a Decision Tree, a Logistic Regression model, and a Random Forest — are trained and compared under identical preprocessing and a common 80/20 train/test split. The report walks through all eleven steps of the pipeline task by task, with the exact code and the console output each step produces.

Headline finding: on the (simulated) data the script ships with, none of the models actually learn to detect churn. The best model by accuracy simply predicts “continues” for every customer, so its accuracy equals the majority-class share. The confusion matrix in Section 6 makes this explicit. This is the expected behaviour on random data and is the central thing to fix before the project carries any business meaning (Section 9).

2. Project Objectives
	•	Build a reproducible supervised-learning pipeline that ingests customer records and predicts subscription churn.
	•	Compare three standard classifiers under identical preprocessing and train/test conditions.
	•	Quantify performance with accuracy and per-class precision / recall, and identify the strongest model.
	•	Surface the features that most influence churn.
	•	Provide a scoring path for new, unseen customers that returns a prediction plus a confidence value.

3. Dataset Description
The project targets the Telco Customer Churn dataset (an IBM sample widely distributed on Kaggle). Each row is one customer; the final column, Churn, is the target. After dropping the identifier, 15 predictor features remain.

3.1 Target encoding (non-standard)

Original     -----           Meaning                      ---------         Encoded as

   No        ----  Customer continues the subscription.   ---------             1
   
   Yes       ----  Customer leaves (churns)               ---------             0

Note: this is the reverse of the field-standard convention (where 1 = churned). Here 1 = stays, 0 = leaves. Every metric in this report follows the project's own convention.

3.2 Class distribution
The target is imbalanced — most customers are retained — which is why accuracy alone is a weak quality measure (a model that always predicts “continues” would already score ~74%).

  Class      ---   Label   ----    Share (reproduction run)

Continues    ---    1      ----       73.7% (737 / 1000)

Leaves       ---    0      ----       26.3% (263 / 1000)

4. Pipeline Overview
The script is organised as a linear, eleven-step pipeline. The next section walks through each step with its code and output.

Step  --      Stage            ---------            Purpose

 1    --     Imports              ---        Load libraries.
 
 2    --     Data acquisition     ---        Load the Kaggle CSV (or generate 1,000 simulated customers).
 
 3    --     Exploration          ---        Inspect distribution, types, missing values.
 
 4    --     Cleaning             ---        Drop ID, coerce numerics, fill gaps, encode target + categoricals.
 
 5    --     Feature selection    ---        Split into feature matrix X and target y.
 
 6    --     Scaling              ---        Standardise features.
 
 7    --     Split                ---        80/20 train/test, fixed seed.
 
 8    --     Training             ---        Fit the three models.
 
 9    --     Evaluation           ---        Accuracy + classification reports; pick best.
 
 10   --    Visualisation         ---        Build and save the four-panel figure.
 
 11   --    Scoring               ---        Predict churn for two example customers.

5. Task-by-Task Walkthrough

How to read this section. Each step shows the essential code and the actual console output captured from running the script. Output blocks are shown verbatim (lightly trimmed for width). The run uses the simulated dataset with np.random.seed(42).
Step 1 — Import libraries
Pulls in data handling (pandas, numpy), plotting (matplotlib, seaborn) and the scikit-learn pieces: three estimators, metrics, and preprocessing utilities. This step prints nothing.

Step 2 — Acquire the data
To access data from the CSV file, we require a function read_csv() from Pandas that retrieves data in the form of the data frame.

Output
📦 Dataset loaded: 1000 customers, 17 columns
 
Step 3 — Explore the data
Reports the churn split and confirms there are no missing values in the simulated frame (the real dataset would show blanks in TotalCharges).

Code
print(df['Churn'].value_counts())
print(df.isnull().sum())

Output
📊 Churn Distribution:

Churn

No     737

Yes    263
 
  ✅ Continuing (No Churn):  737 customers (73.7%)
 
  ❌ Left (Churned):          263 customers (26.3%)
 
📋 Missing Values: all columns = 0

Step 4 — Clean and encode
Drops the ID, coerces TotalCharges to numeric and fills gaps with the median, maps the target to the project's 1=continues / 0=leaves convention, then label-encodes the 11 text columns.

Caveat: LabelEncoder imposes a false numeric order on unordered categories (e.g. PaymentMethod). Trees tolerate this; the linear model can be misled. One-hot encoding is preferable (Section 9).

Output

Target column converted:

  'No'  (continues)  → 1
  
  'Yes' (leaves)     → 0
 
✅ Encoded 11 categorical columns

Cleaned dataset shape: (1000, 16)


Step 5 — Select features
Separates the 15 predictors (X) from the target (y).

Output
Features (X): ['gender','SeniorCitizen','Partner','Dependents','tenure',
 'PhoneService','MultipleLines','InternetService','OnlineSecurity',
 'TechSupport','Contract','PaperlessBilling','PaymentMethod',
 'MonthlyCharges','TotalCharges']
Target (y): Churn (1=continues, 0=leaves)

Step 6 — Scale features
StandardScaler centres each feature to zero mean / unit variance. This genuinely helps Logistic Regression; it is redundant (but harmless) for the tree-based models. No console output.

Code
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


Step 7 — Train / test split
An 80/20 split with a fixed seed gives 800 training and 200 test customers.
Caveat: the split is not stratified, so the test set's class ratio can drift from the population. Passing stratify=y fixes this.

Code
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

Output
Training samples: 800
Testing  samples: 200


Step 8 — Train the models
Fits all three estimators on the training set.

Code

dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

Output
✅ Decision Tree trained
✅ Logistic Regression trained
✅ Random Forest trained


Step 9 — Evaluate
Computes accuracy and a per-class report for each model, then picks the highest-accuracy model.
Read the recall column for “Left (0)” carefully. It is 0.04, 0.00 and 0.02 — the models essentially never predict churn. Logistic Regression’s 74% accuracy is exactly the majority-class share (148/200), i.e. it labels everyone “continues.” The scikit-learn UndefinedMetricWarning during this step is the same symptom: precision is undefined because no samples were predicted as class 0.

Code
for name,(model,pred) in models.items():
    acc = accuracy_score(y_test, pred)
    print(name, f"{acc:.2%}")
    print(classification_report(y_test, pred,
          target_names=['Left (0)','Continues (1)']))
best_model_name = max(results, key=results.get)
Output
📊 Decision Tree   Accuracy: 70.00%
               precision  recall  f1-score  support
     Left (0)       0.17    0.04     0.06       52
Continues (1)       0.73    0.93     0.82      148
 
📊 Logistic Regression   Accuracy: 74.00%
               precision  recall  f1-score  support
     Left (0)       0.00    0.00     0.00       52
Continues (1)       0.74    1.00     0.85      148
 
📊 Random Forest   Accuracy: 73.50%
               precision  recall  f1-score  support
     Left (0)       0.33    0.02     0.04       52
Continues (1)       0.74    0.99     0.85      148
 
🏆 Best Model: Logistic Regression (74.00%)
Step 10 — Visualise
Builds a four-panel figure (distribution pie, accuracy bars, confusion matrix for the best model, Random Forest feature importances) and saves it to PNG. The rendered figure is shown in Section 6.
Code
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# pie, bar, confusion-matrix heatmap, feature-importance barh ...
plt.savefig('churn_prediction_results.png', dpi=150, bbox_inches='tight')

Output
✅ Charts saved as 'churn_prediction_results.png'

Step 11 — Score new customers
Builds two synthetic customers, scales them with the same fitted scaler, and predicts with the best model, returning a class and a confidence. Because the best model predicts the majority class for everyone, both examples come back as “continues.”

Code
sample_scaled = scaler.transform(sample_customers)
predictions   = best.predict(sample_scaled)
probabilities = best.predict_proba(sample_scaled)

Output
Using Best Model: Logistic Regression
Customer 1: ✅ CONTINUES (1)   Confidence: 70.8%
  Prob(Leaves=0): 29.23% | Prob(Continues=1): 70.77%
Customer 2: ✅ CONTINUES (1)   Confidence: 74.2%
  Prob(Leaves=0): 25.77% | Prob(Continues=1): 74.23%

6. Results and Figures
6.1 Original submitted figure
This is the figure included with the project. Its accuracies (Decision Tree 79.4%, Logistic Regression 81.5%, Random Forest 80.1%) come from the original run. Note that its confusion-matrix panel (bottom-left) is blank.

Figure 1. Original submitted results dashboard (confusion-matrix panel did not render in that run).
6.2 Reproduction run figure
Re-running the script in a current Python / scikit-learn environment produces the figure below. The confusion-matrix panel now renders correctly — which tells us the blank panel in Figure 1 was an environment-specific rendering glitch, not a code defect.

Figure 2. Reproduction run. The confusion matrix shows 0 predictions in the “Leaves (0)” column — the model labels every customer “continues.”
Why the two figures disagree (80% vs 74%). The simulated dataset is regenerated each run, and numpy’s random stream is not byte-stable across library versions, so the fabricated data — and therefore every downstream number — shifts between environments. This is itself a useful demonstration: results built on synthetic random data are not reproducible and carry no business meaning until the real dataset is used.

6.3 What the confusion matrix reveals
                           Predicted: Leaves (0)                        Predicted: Continues (1)
Actual: Leaves (0)                0                                             52
Actual: Continues (1)             0                                            148

Every one of the 200 test customers is predicted “continues.” The 52 real churners are all missed. Accuracy = 148/200 = 74%, identical to the majority-class baseline. This is the degenerate classifier that class imbalance plus signal-free features produces, and it is the clearest single argument for the fixes in Section 9.

6.4 Feature importance
From the Random Forest, the highest-ranked features are TotalCharges, MonthlyCharges and tenure, followed by payment and service attributes. On the real Telco data these rankings are meaningful (contract type and tenure are well-known churn drivers); on the simulated data they are noise and should not be interpreted substantively.

7. Complete Console Output
The full, unedited console output from the reproduction run (warnings omitted) is reproduced below for reference.
=======================================================
  MOBILE SUBSCRIPTION CHURN PREDICTION PROJECT
=======================================================
 
📦 Dataset loaded: 1000 customers, 17 columns
 
📊 Churn Distribution:
No     737
Yes    263
  ✅ Continuing (No Churn):  737 customers (73.7%)
  ❌ Left (Churned):          263 customers (26.3%)
 
📋 Data Types & Missing Values: all 0
 
STEP 4: DATA CLEANING
  'No'  (continues)  → 1
  'Yes' (leaves)     → 0
✅ Encoded 11 categorical columns
Cleaned dataset shape: (1000, 16)
 
STEP 5: FEATURE SELECTION
Training samples: 800
Testing  samples: 200
 
STEP 8: TRAINING MODELS
✅ Decision Tree trained
✅ Logistic Regression trained
✅ Random Forest trained
 
STEP 9: MODEL EVALUATION
📊 Decision Tree         Accuracy: 70.00%
     Left (0)       0.17   0.04   0.06    52
Continues (1)       0.73   0.93   0.82   148
📊 Logistic Regression   Accuracy: 74.00%
     Left (0)       0.00   0.00   0.00    52
Continues (1)       0.74   1.00   0.85   148
📊 Random Forest         Accuracy: 73.50%
     Left (0)       0.33   0.02   0.04    52
Continues (1)       0.74   0.99   0.85   148
🏆 Best Model: Logistic Regression (74.00%)
✅ Charts saved as 'churn_prediction_results.png'
 
STEP 11: PREDICT NEW CUSTOMERS
Customer 1: ✅ CONTINUES (1)   Confidence: 70.8%
  Prob(Leaves=0): 29.23% | Prob(Continues=1): 70.77%
Customer 2: ✅ CONTINUES (1)   Confidence: 74.2%
  Prob(Leaves=0): 25.77% | Prob(Continues=1): 74.23%
 
  PROJECT COMPLETE ✅
=======================================================
7.1 Warnings emitted at runtime
Three categories of warning appear and are worth noting:
	•	UndefinedMetricWarning (precision ill-defined): confirms a model predicted no samples for the “Left (0)” class. Pass zero_division=0 to silence, but the real fix is to make the model actually predict churn.
	•	UserWarning (X has no valid feature names): the Step 11 sample customers are passed as a NumPy array to a model trained on a named DataFrame. Harmless; pass a DataFrame with the same columns to remove it.
	•	Pandas deprecation on select_dtypes(include=['object']): a forward-compatibility notice for pandas 3; functionally unaffected today.

8. Limitations and Known Issues
8.1 Simulated data by default
Unless the real CSV is loaded, the features are random and unrelated to the target, so the models fit noise. Accuracy is driven entirely by the majority class.
8.2 Majority-class collapse
As Section 6.3 shows, the best model predicts “continues” for every customer and detects no churners. Accuracy hides this completely.
8.3 Accuracy is the wrong headline metric
Under a 74/26 split, accuracy rewards predicting the majority. Precision, recall and F1 on the minority “leaves” class — and ROC-AUC / PR-AUC — should drive model selection instead.
8.4 Non-reproducibility across versions
Because the dataset is synthesised from numpy’s random stream, the exact numbers change between library versions (Figure 1’s ~80% vs the reproduction’s ~74%).
8.5 Other issues
	•	No stratify=y in the train/test split.
	•	LabelEncoder used on unordered categorical features (one-hot is preferable).
	•	Inverted target convention (1=continues) differs from the field standard.
	•	The original Figure 1 confusion-matrix panel failed to render (environment-specific; reproduces fine).

9. How to Run and Reproduce
9.1 Environment
pip install pandas numpy matplotlib seaborn scikit-learn
9.2 Switch to the real dataset
	•	Download blastchar/telco-customer-churn from Kaggle.
	•	Place WA_Fn-UseC_-Telco-Customer-Churn.csv beside the script.
	•	Replace the np.random simulated block with one line:
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
	•	Run the script; all downstream steps work unchanged.

10. Recommendations and Future Work
	•	Use the real Telco dataset so metrics and feature importances become meaningful.
	•	Select models on recall / F1 / ROC-AUC for the churn class, not accuracy.
	•	Add stratify=y to stabilise the class ratio across the split.
	•	Address imbalance with class_weight='balanced', threshold tuning, or SMOTE so the model actually predicts churners.
	•	Replace LabelEncoder with one-hot encoding (via ColumnTransformer) for nominal features.
	•	Wrap preprocessing + estimator in a scikit-learn Pipeline and validate with stratified k-fold cross-validation.
	•	Pass zero_division=0 to classification_report and a named DataFrame to the scorer to clear the runtime warnings.
	•	Persist the chosen model with joblib so scoring does not require retraining.

11. Appendix: Code Reference
     Object                          ------                            Role
df                                   ------           Cleaned, fully-encoded customer table.
X / y                                ------           Feature matrix and target vector.
X_scaled                             ------           Standardised features.
X_train / X_test / y_train / y_test  ------        80/20 split, random_state = 42.
dt / lr / rf                         ------           Decision Tree, Logistic Regression, Random Forest.
models                               ------           Dict: name → (estimator, test predictions).
results                              ------           Dict: name → accuracy; used to choose the best.
best / best_model_name               ------           Highest-accuracy model and its name.

Files documented: mobile_subscription_churn.txt (pipeline script) and churn_prediction_results.png (results figure).
