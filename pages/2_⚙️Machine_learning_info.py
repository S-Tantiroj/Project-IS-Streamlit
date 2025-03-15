import streamlit as st

st.set_page_config(
    page_title="Machine learning",
    page_icon="⚙️",
)

st.title("Machine learning information")


st.subheader("1. การเตรียมข้อมูล (Data Preparation)")

st.write("""
เพื่อให้ได้โมเดลที่มีประสิทธิภาพสูงสุด จำเป็นต้องมีการเตรียมข้อมูล ดังนี้:
1. **โหลดข้อมูลจาก Dataset**: ข้อมูลที่ใช้ในการพัฒนาโมเดลนี้มาจาก Kaggle: [Salary Data](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data)
2. **ล้างข้อมูล (Data Cleaning):** ลบค่าที่หายไป (`NaN`) และตรวจสอบค่าผิดปกติ
3. **แปลงข้อมูลหมวดหมู่ (Categorical Encoding):** ใช้ `LabelEncoder` แปลงข้อมูลที่เป็นข้อความเป็นตัวเลข
4. **ปรับขนาดข้อมูล (Feature Scaling):** ใช้ `StandardScaler` เพื่อทำให้ค่าทุกคอลัมน์มีขนาดที่สมดุลกัน
""")

st.subheader("2. ทฤษฎีของอัลกอริทึมที่พัฒนา")

st.write("""
เราได้ทดสอบโมเดล Machine Learning 3 ตัว และเลือกโมเดลที่ดีที่สุด ดังนี้:

1. **Random Forest Regressor** 🌲
   - เป็นการรวมหลาย Decision Trees เพื่อลด Overfitting
   - ใช้ Bootstrap Aggregation (Bagging) ทำให้โมเดลมีความแม่นยำสูง

2. **Gradient Boosting Regressor** 📈
   - ใช้วิธี Boosting ซึ่งเรียนรู้จากข้อผิดพลาดของโมเดลก่อนหน้า
   - ปรับปรุงความแม่นยำ แต่ใช้เวลาฝึกโมเดลมากกว่า

3. **XGBoost Regressor** ⚡
   - เป็นเวอร์ชันที่เร็วและมีประสิทธิภาพสูงของ Gradient Boosting
   - ใช้เทคนิค Regularization เพื่อลด Overfitting

**🔬 ผลลัพธ์:**
เราพบว่า **Random Forest** ให้ผลลัพธ์ดีที่สุด ดังนั้นจึงเลือกใช้เป็นโมเดลหลักของเรา
""")

st.subheader("3. ขั้นตอนการพัฒนาโมเดล")

st.write("""
1. **โหลดและล้างข้อมูล** จาก `Salary Data.csv (ซึ่งดาวน์โหลดมาจาก Kaggle)`
2. **แปลงข้อมูลข้อความเป็นตัวเลข** ด้วย `LabelEncoder`
3. **แบ่งข้อมูลเป็นชุด Train/Test** (80/20)
4. **ปรับขนาดข้อมูล** ด้วย `StandardScaler`
5. **ฝึกโมเดลทั้ง 3 ตัว** และประเมินผลลัพธ์ด้วย `R², MAE, RMSE`
6. **เลือกโมเดลที่ดีที่สุด** และบันทึกเป็น `best_model.pkl`
7. **นำโมเดลมาใช้ใน Streamlit App** เพื่อให้สามารถพยากรณ์เงินเดือนแบบอินเทอร์แอคทีฟได้

""")

st.subheader("สรุปผล (Conclusion)")
st.write("""
จากการทดลอง เราพบว่า **Random Forest Regressor** ให้ผลลัพธ์ดีที่สุด

แอปของเราสามารถช่วยให้ผู้ใช้คาดการณ์เงินเดือนได้จากคุณสมบัติต่าง ๆ เช่น อายุ, เพศ, ระดับการศึกษา, ตำแหน่งงาน และประสบการณ์ทำงาน
""")
