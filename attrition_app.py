import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_and_train():
    df = pd.read_csv("https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv")
    df = df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_cols = load_and_train()

# Header
st.title("👔 Employee Attrition Predictor")
st.markdown("### Predict whether an employee is likely to leave the company")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("👤 Employee Profile")

age = st.sidebar.slider("Age", 18, 60, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4, 5], format_func=lambda x: {1:"Below College", 2:"College", 3:"Bachelor", 4:"Master", 5:"Doctor"}[x])

st.sidebar.header("💼 Job Details")
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
job_level = st.sidebar.slider("Job Level (1-5)", 1, 5, 2)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)
business_travel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

st.sidebar.header("💰 Compensation")
monthly_income = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000)
daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, 500)
hourly_rate = st.sidebar.slider("Hourly Rate", 30, 100, 60)
percent_salary_hike = st.sidebar.slider("Salary Hike (%)", 0, 25, 10)
stock_option_level = st.sidebar.slider("Stock Option Level (0-3)", 0, 3, 1)

st.sidebar.header("🏢 Work Environment")
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
work_life_balance = st.sidebar.slider("Work Life Balance (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
distance_from_home = st.sidebar.slider("Distance From Home (km)", 1, 30, 5)

st.sidebar.header("📅 Experience")
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 18, 3)
years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
years_with_curr_manager = st.sidebar.slider("Years With Current Manager", 0, 17, 3)
num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 9, 2)
training_times_last_year = st.sidebar.slider("Training Times Last Year", 0, 6, 3)
performance_rating = st.sidebar.slider("Performance Rating (1-4)", 1, 4, 3)

# Encode categoricals
gender_val = 1 if gender == "Male" else 0
overtime_val = 1 if overtime == "Yes" else 0
travel_map = {"Non-Travel": 0, "Travel_Rarely": 2, "Travel_Frequently": 1}
travel_val = travel_map[business_travel]
dept_map = {"Human Resources": 0, "Research & Development": 1, "Sales": 2}
dept_val = dept_map[department]
role_map = {"Healthcare Representative": 0, "Human Resources": 1, "Laboratory Technician": 2,
            "Manager": 3, "Manufacturing Director": 4, "Research Director": 5,
            "Research Scientist": 6, "Sales Executive": 7, "Sales Representative": 8}
role_val = role_map[job_role]
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
marital_val = marital_map[marital_status]

input_data = pd.DataFrame([[
    age, travel_val, daily_rate, dept_val, distance_from_home,
    education, 0, gender_val, hourly_rate, job_involvement,
    job_level, role_val, job_satisfaction, marital_val, monthly_income,
    0, num_companies_worked, overtime_val, percent_salary_hike,
    performance_rating, relationship_satisfaction, stock_option_level,
    total_working_years, training_times_last_year, work_life_balance,
    years_at_company, years_in_current_role, years_since_last_promotion,
    years_with_curr_manager, environment_satisfaction
]], columns=feature_cols)

# Main content
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Monthly Income", f"${monthly_income:,}")
with col2:
    st.metric("Job Satisfaction", f"{job_satisfaction}/4")
with col3:
    st.metric("Years at Company", f"{years_at_company} yrs")

st.markdown("---")

if st.button("🔍 Predict Attrition Risk", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("## Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — This employee is likely to leave")
        st.progress(int(probability[1] * 100))
        st.markdown(f"**Attrition Probability: {probability[1]*100:.1f}%**")
    else:
        st.success(f"✅ LOW RISK — This employee is likely to stay")
        st.progress(int(probability[0] * 100))
        st.markdown(f"**Retention Probability: {probability[0]*100:.1f}%**")

    st.markdown("---")
    st.markdown("### Key Risk Factors")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🕐 OverTime: {'Yes' if overtime_val == 1 else 'No'}")
        st.info(f"😊 Job Satisfaction: {job_satisfaction}/4")
        st.info(f"⚖️ Work Life Balance: {work_life_balance}/4")
    with col2:
        st.info(f"📍 Distance From Home: {distance_from_home} km")
        st.info(f"📈 Salary Hike: {percent_salary_hike}%")
        st.info(f"🏢 Years at Company: {years_at_company}")