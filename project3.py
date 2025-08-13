import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  

df = pd.read_csv("processed_employee_data.csv")

numerical_cols = [
    "age", "daily_rate", "distance_from_home", "education", "employee_count",
    "employee_number","hourly_rate",
    "job_level", "job_satisfaction", "monthly_income", "monthly_rate",
    "num_companies_worked", "percent_salary_hike", "performance_rating",
     "standard_hours", "stock_option_level",
    "total_working_years",  "work_life_balance","years_at_company", 
    "years_in_current_role", "years_since_last_promotion",
     "performance_score", "job_hopper",
     "frequent_overtime"
]

def main():
    st.title(" Employee Attrition Prediction Dashboard")
    st.sidebar.header(" Employee Features")

    user_input = {}

    education_options = ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]
    business_travel_options = ["Travel_Frequently", "Travel_Rarely"]
    department_options = ["Research & Development", "Sales"]
    marital_status_options = ["Married", "Single"]
    tenure_options = ["Mid-Level", "New", "Senior"]

    user_input["education_field"] = st.sidebar.selectbox("Education Field", education_options)
    user_input["business_travel"] = st.sidebar.selectbox("Business Travel", business_travel_options)
    user_input["department"] = st.sidebar.selectbox("Department", department_options)
    user_input["marital_status"] = st.sidebar.selectbox("Marital Status", marital_status_options)
    user_input["tenure_category"] = st.sidebar.selectbox("Tenure Category", tenure_options)

    job_role_options = df["job_role"].dropna().unique().tolist()
    job_role_selection = st.sidebar.selectbox("Job Role", job_role_options)

    try:
        job_role_encoder = joblib.load("job_role_encoder.pkl")
    except FileNotFoundError:
        job_role_encoder = LabelEncoder()
        df["job_role"] = job_role_encoder.fit_transform(df["job_role"])
        joblib.dump(job_role_encoder, "job_role_encoder.pkl")

    
    if set(job_role_options) - set(job_role_encoder.classes_):
        job_role_encoder = LabelEncoder()
        df["job_role"] = job_role_encoder.fit_transform(df["job_role"])
        joblib.dump(job_role_encoder, "job_role_encoder.pkl")

    if job_role_selection in job_role_encoder.classes_:
        user_input["job_role"] = job_role_encoder.transform([job_role_selection])[0]
    else:
        st.warning(f"'{job_role_selection}' not in training data. Assigning default value.")
        user_input["job_role"] = -1  

    categorical_features = {
        "education_field": education_options,
        "business_travel": business_travel_options,
        "department": department_options,
        "marital_status": marital_status_options,
        "tenure_category": tenure_options
    }

    for category, options in categorical_features.items():
        for option in options:
            col_name = f"{category}_{option}"  
            user_input[col_name] = 1 if user_input[category] == option else 0

    
    for category in categorical_features.keys():
        del user_input[category]

    
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            min_val, max_val, mean_val = df[col].min(), df[col].max(), df[col].mean()
            user_input[col] = st.sidebar.number_input(
                f"{col}", min_value=float(min_val), max_value=float(max_val), value=float(mean_val)
            )
        else:
            user_input[col] = 0  

    input_df = pd.DataFrame([user_input])


    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0


    input_df = input_df[feature_names]

    st.write(" **Processed Input Data:**")
    st.dataframe(input_df)

    input_scaled = scaler.transform(input_df)

    if st.sidebar.button("Enter"):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            st.error(f" **Attrition Prediction: YES** - Probability: {probability:.2f}")
        else:
            st.success(f" **Attrition Prediction: NO** - Probability: {probability:.2f}")

if __name__ == "__main__":
    main()
