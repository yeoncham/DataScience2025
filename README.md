# DataScience2025 Project  
## Thyroid Cancer Classification: Malignant vs. Benign

---

## 🎯 Goal  
To improve model performance in classifying thyroid cancer as **malignant or benign** using a **Random Forest classifier**.

---

## ⚙️ Preprocessing Strategy  
- Used an **XGBoost classifier** to measure feature importance.
- Removed features with low importance scores.
- Although some low-importance features may contribute at the **leaf level**, due to time constraints, we prioritized feature reduction based on importance scores.

---

## 🔄 Handling Data Imbalance  
- Faced a **class imbalance** issue at the beginning.
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to synthesize minority class samples.
- ⚠️ Note: SMOTE only balances the **training set**, not the **test set**. Addressing test set imbalance is a remaining challenge.

---

## 🔍 Model Evaluation: 10-Fold Stratified Cross-Validation  

### 1. 📊 Base Model vs. Model with Derived Features  
- Evaluation via 10-fold stratified cross-validation  
- ✅ **Accuracy improvement**: +1.9 percentage points (pp)

### 2. 🔧 Derived Features vs. Derived Features + Hyperparameter Tuning  
- Used **GridSearchCV** for tuning
- ✅ **Accuracy improvement**: +0.6pp

### 3. 🔁 Base Model vs. Final Tuned Model  
- Compared performance on the **entire dataset**
- ✅ **Total accuracy gain**: +2.33pp

---

## 📈 Statistical Validation: Paired T-Test  
- Conducted a **paired t-test** to verify statistical significance
- ✅ **p-value < 0.05**, indicating significant improvement

---

## 📊 Hypothesis Testing: Chi-Square Test  
- Tested statistical association of derived features with the target label
- ⚠️ Note: Data analysis reveals **correlation**, not **causation**

---

## 💡 Reflections & Challenges  

### 1. Data Imbalance  
- Solved partially with SMOTE  
- **Test set imbalance** still remains an issue

### 2. Lack of Medical Domain Knowledge  
- All team members are **computer science majors**
- Focused on **data-driven insights** rather than domain-specific causal interpretations

---

## ✅ Summary  

| Step                          | Description                                | Accuracy Gain |
|-------------------------------|--------------------------------------------|----------------|
| Feature Engineering           | Selected features via XGBoost              | +1.9pp         |
| Hyperparameter Tuning         | GridSearchCV on derived model              | +0.6pp         |
| Combined Effect               | Final model vs base model                  | **+2.33pp**    |

---

## 📌 Future Work Suggestions  
- Explore **advanced resampling techniques** (e.g., SMOTE+ENN, ADASYN)
- Collaborate with **domain experts** for feature interpretation
- Apply **explainable AI methods** such as **SHAP** or **LIME**
