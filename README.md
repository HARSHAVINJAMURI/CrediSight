# 🧠 CrediSight – Credit Card Fraud Detection

**CrediSight** is an interactive **Streamlit-based Machine Learning application** designed to analyze, visualize, and predict patterns in **credit card application and fraud detection data**. It enables analysts and financial institutions to quickly assess customer risk, detect anomalies, and gain actionable insights.

---

## 🚀 Features

### 🔹 1. Data Input Options

* **Upload your own dataset (CSV)** or use the **built-in sample dataset**.
* Automatically detects and preprocesses features for analysis.

### 🔹 2. Data Preprocessing

* Encodes categorical variables automatically using `LabelEncoder`.
* Derives additional features such as:

  * `total_delay` (sum of document delay and processing days)
  * `dpd_bucket` (bucketized delay duration)

### 🔹 3. Visualization Suite

Easily explore trends and relationships with multiple visualization options:

* **Histogram** – Distribution of numeric variables.
* **Boxplot** – Compare numeric vs categorical variables.
* **Correlation Heatmap** – Explore feature correlations.
* **Countplot** – Frequency of categorical variables.
* **Scatter Plot** – Visualize relationships and clusters.
* **Map View** – Plot customer locations (latitude/longitude).

### 🔹 4. Machine Learning Modules

Perform supervised and unsupervised learning tasks directly within the app:

#### 🧩 Application Status Classification

* Predicts if a credit card application will be **Approved**, **Rejected**, or **Pending**.
* Model: `RandomForestClassifier`
* Metrics: Classification report & confusion matrix.

#### ⚠️ Default Flag Prediction

* Predicts the likelihood of a customer defaulting.
* Model: `GradientBoostingClassifier`
* Metrics: Classification report, ROC Curve, AUC score.

#### ⏳ Processing Days Prediction (Regression)

* Estimates how long an application will take to process.
* Model: `RandomForestRegressor`
* Metrics: Mean Absolute Error (MAE), R² Score, Actual vs Predicted plot.

#### 📦 Delay Bucket Classification

* Classifies applications into delay ranges (e.g., `0-30`, `30-60`, etc.).
* Model: `XGBClassifier`
* Visualization: Feature importance plot.

#### 🚨 Anomaly Detection

* Detects potentially suspicious or fraudulent applications.
* Model: `IsolationForest`
* Visualization: Countplot and scatter plot (e.g., Credit Score vs Salary).

---

## 🧮 Tech Stack

| Component            | Description                        |
| -------------------- | ---------------------------------- |
| **Frontend/UI**      | [Streamlit](https://streamlit.io/) |
| **Data Handling**    | Pandas, NumPy                      |
| **Machine Learning** | Scikit-learn, XGBoost              |
| **Visualization**    | Matplotlib, Seaborn                |

---

## 📊 Sample Dataset

The app comes with a preloaded sample dataset that simulates real-world credit card applications with the following fields:

| Column                      | Description                                   |
| --------------------------- | --------------------------------------------- |
| `customer_id`               | Unique customer identifier                    |
| `age`                       | Applicant age                                 |
| `salary`                    | Annual income                                 |
| `credit_score`              | Creditworthiness score                        |
| `employment_type`           | Type of employment (Salaried / Self-employed) |
| `application_source`        | Channel of application (Web / Agent / Branch) |
| `document_completeness`     | % of documents submitted correctly            |
| `past_payment_delays`       | Number of previous delays                     |
| `credit_utilization`        | Ratio of credit usage                         |
| `document_submission_delay` | Days taken to submit documents                |
| `processing_days`           | Days taken to process the application         |
| `application_status`        | Approved / Rejected / Pending                 |
| `default_flag`              | 1 = Defaulted, 0 = Non-default                |
| `latitude`, `longitude`     | Geolocation coordinates                       |

---

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## Usage

1. Choose your data source: Upload CSV or use the sample data.
2. Preprocess and explore the data using visualizations.
3. Select a Machine Learning task from the sidebar:

   * Application Status Classification
   * Default Flag Prediction
   * Processing Days Prediction
   * Delay Bucket Classification
   * Anomaly Detection
4. View model metrics, plots, and predictions interactively.


## Requirements

All required packages are listed in `requirements.txt`:

```
streamlit==1.39.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.1
matplotlib==3.9.2
seaborn==0.13.2
```
---

## 🧠 Project Workflow

1. **Load Data** → Upload or use built-in dataset.
2. **Preprocess Data** → Handle categorical encoding and derived features.
3. **Visualize** → Explore numeric and categorical trends.
4. **Select ML Task** → Choose prediction or anomaly detection mode.
5. **Train & Evaluate** → Run models, view metrics and plots instantly.

---

## 🧰 Folder Structure

```
📦 CrediSight
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies list
├── README.md                   # Project documentation
└── sample_data.csv             # Optional example data
```

---

## 🧑‍💻 Author

**Developed by:** Harsha Vinjamuri (Data Science & ML Enthusiast)
**Purpose:** Demonstration of end-to-end credit risk modeling and anomaly detection using modern ML tools.

---

## 📄 License

This project is released under the **MIT License** — free to use and modify for educational or commercial purposes.

---

## 💡 Future Enhancements

* Add **SHAP-based explainability** for feature impact.
* Integrate **live fraud alert dashboard**.
* Connect with **real-time APIs** for dynamic financial data.
* Deploy on **Streamlit Cloud / AWS / Azure** for production use.

---

**🎯 CrediSight — Empowering Smarter, Safer Credit Decisions with AI.**

