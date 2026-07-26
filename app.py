import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="DSPristine - HR Analytics", layout="wide")

st.title("💼 DSPristine HRMS: Employee Retention Predictor")
st.write("Real-time-la employee leave aaga vaaippu irukkaa nu check pannunga.")
st.markdown("---")

# AI Brain-a Load Panrom
@st.cache_resource
def load_model():
    with open('dsp_retention_model.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_model()
model = pipeline['model']
encoders = pipeline['encoders']
features = pipeline['features']

# Input Fields (Left Side)
st.sidebar.header("📋 Employee Details")
monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=1000, value=5000)
overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])
total_working_years = st.sidebar.slider("Total Working Experience", 0, 40, 8)
years_at_company = st.sidebar.slider("Years at Company", 0, 30, 4)
years_since_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)

# Predict Button
if st.button("🚀 Analyze Exit Risk", type="primary"):
    input_dict = {col: 0 for col in features}
    input_dict['MonthlyIncome'] = monthly_income
    input_dict['TotalWorkingYears'] = total_working_years
    input_dict['YearsAtCompany'] = years_at_company
    input_dict['YearsSinceLastPromotion'] = years_since_promotion
    input_dict['JobSatisfaction'] = job_satisfaction
   
    if 'OverTime' in encoders and overtime in encoders['OverTime'].classes_:
        input_dict['OverTime'] = encoders['OverTime'].transform([overtime])[0]

    input_df = pd.DataFrame([input_dict])
    risk_score = model.predict_proba(input_df)[0][1]

    # Result
    st.subheader("Results:")
    st.metric(label="Exit Probability Score", value=f"{risk_score * 100:.1f}%")
   
    if risk_score >= 0.50:
        st.error("🚨 Status: HIGH RISK (Likely to Leave)")
        st.write("💡 **HR Action Plan:** Overtime control, Promotion review & Salary increment schedule pannunga.")
    else:
        st.success("✅ Status: LOW RISK (Likely to Stay)")