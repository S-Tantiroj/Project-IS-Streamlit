import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

st.set_page_config(
    page_title="Machine learning Demo",
    page_icon="💵",
)

st.title("💰 Salary prediction")
st.write("ป้อนข้อมูลด้านล่างเพื่อทำนายเงินเดือน")

# กำหนดพาธไปยังโฟลเดอร์ที่เก็บโมเดล
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Machine_learning"))
model_path = os.path.join(base_dir, "best_model.pkl")
encoder_path = os.path.join(base_dir, "label_encoders.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")

# ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
for path in [model_path, encoder_path, scaler_path]:
    if not os.path.exists(path):
        st.error(f"❌ ไม่พบไฟล์: {path}")
        st.stop()


# โหลดโมเดลและตัวเข้ารหัส
best_model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)

# สร้างช่องให้ผู้ใช้ป้อนข้อมูล
age = st.number_input("อายุ (Age)", min_value=18, max_value=100, value=30)
gender = st.selectbox("เพศ (Gender)", label_encoders["Gender"].classes_)
education = st.selectbox("ระดับการศึกษา (Education Level)", label_encoders["Education Level"].classes_)
job_title = st.selectbox("ตำแหน่งงาน (Job Title)", label_encoders["Job Title"].classes_)
years_experience = st.number_input("ประสบการณ์ทำงาน (Years of Experience)", min_value=0, max_value=50, value=5)

# ปุ่มกดเพื่อทำนาย
if st.button("ทำนายเงินเดือน"):
    # แปลงค่าที่เป็นหมวดหมู่ให้เป็นตัวเลข
    gender_encoded = label_encoders["Gender"].transform([gender])[0]
    education_encoded = label_encoders["Education Level"].transform([education])[0]
    job_encoded = label_encoders["Job Title"].transform([job_title])[0]
    
    # เตรียมข้อมูลสำหรับโมเดล
    input_data = np.array([[age, gender_encoded, education_encoded, job_encoded, years_experience]])
    input_scaled = scaler.transform(input_data)
    
    # ทำนายเงินเดือนจากโมเดลที่ดีที่สุด
    salary_pred = best_model.predict(input_scaled)[0]
    
    # แสดงผลลัพธ์
    st.subheader("💵")
    st.write(f"**ค่าประมาณเงินเดือนที่ทำนายได้:** ${salary_pred:,.2f}")