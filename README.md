# 🫀 Heart Disease Prediction – Full Machine Learning Pipeline

## 🚀 Project Overview
This project builds a **complete machine learning pipeline** to analyze and predict heart disease risk using the **UCI Heart Disease dataset**.  
It covers the **entire workflow** from data cleaning and preprocessing to feature selection, model training, evaluation, and deployment with an interactive web app.

The project demonstrates the power of **supervised and unsupervised learning**, combined with **dimensionality reduction (PCA)**, **hyperparameter tuning**, and **visualizations** for insightful exploration.

---

## 🎯 Objectives
- Clean and preprocess data (handle missing values, encode categorical features, scale numerical features).  
- Apply **PCA** for dimensionality reduction while retaining key information.  
- Perform **feature selection** using Random Forest importance, RFE, and Chi-Square tests.  
- Train and evaluate **supervised models**: Logistic Regression, Decision Tree, Random Forest, and SVM.  
- Apply **unsupervised learning**: K-Means and Hierarchical Clustering for pattern discovery.  
- Optimize model performance using **GridSearchCV and RandomizedSearchCV**.  
- Build a **Streamlit web interface** for real-time predictions.  
- Optionally deploy publicly using **Ngrok** and host on **GitHub**.

---

## 🛠️ Tools & Technologies
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib  
- **Dimensionality Reduction & Feature Selection:** PCA, RFE, Chi-Square Test  
- **Supervised Models:** Logistic Regression, Decision Tree, Random Forest, SVM  
- **Unsupervised Models:** K-Means, Hierarchical Clustering  
- **Deployment:** Streamlit, Ngrok  

---

## 📊 Random Forest Model Performance
- **Test Accuracy:** 90.16%  
- **Training Accuracy:** 87.60%  

**Classification Report:**  

| Class | Precision | Recall | F1-score | Support |
|-------|----------|--------|----------|--------|
| 0     | 0.94     | 0.88   | 0.91     | 33     |
| 1     | 0.87     | 0.93   | 0.90     | 28     |
| **Overall Accuracy** | - | - | 0.90 | 61 |
| **Macro Avg** | 0.90 | 0.90 | 0.90 | 61 |
| **Weighted Avg** | 0.90 | 0.90 | 0.90 | 61 |

> ⚡ This model achieves **high precision and recall**, demonstrating strong predictive power for heart disease risk.

---

## 🧩 Project Workflow
1. **Data Preprocessing & Cleaning**  
   - Handle missing values, encode categorical variables, standardize numerical features.  
   - Perform exploratory data analysis (EDA) using histograms, correlation heatmaps, and boxplots.

2. **Dimensionality Reduction (PCA)**  
   - Reduce feature dimensions while retaining essential variance.  
   - Visualize variance explained by principal components.

3. **Feature Selection**  
   - Use Random Forest importance and RFE to select key features.  
   - Apply Chi-Square test to ensure statistical significance.

4. **Supervised Learning**  
   - Train Logistic Regression, Decision Tree, Random Forest, and SVM.  
   - Evaluate with accuracy, precision, recall, F1-score, ROC curve, and AUC.

5. **Unsupervised Learning (Clustering)**  
   - Apply K-Means and Hierarchical Clustering to discover patterns.  
   - Compare clusters with actual labels for insights.

6. **Hyperparameter Tuning**  
   - Optimize models using GridSearchCV and RandomizedSearchCV.  
   - Select the best-performing models for deployment.

7. **Deployment & Streamlit App [Bonus]**  
   - Build a **user-friendly Streamlit UI** for real-time predictions.  
   - Optionally share via **Ngrok** for public access.

---


## 📂 How to Run
1. Clone the repository:
```bash
git clone https://github.com/Mahmoud-Elqalini/Full-Pipeline-on-Heart-Disease.git
cd Full-Pipeline-on-Heart-Disease
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```
