import streamlit as st


st.set_page_config(
    page_title="Neural Networks",
    page_icon="🧠",
)

st.title("Neural network information")
st.subheader("การทำงานของโมเดล Card Classifier")
st.write("โมเดลนี้ถูกฝึกมาเพื่อจำแนกไพ่ 53 ใบ โดยใช้ CNN (Convolutional Neural Network) โดยข้อมูลที่ใช้ในการพัฒนาโมเดลนี้มาจาก Kaggle: [Cards Image Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)")

st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
st.write("โมเดลถูกฝึกโดยใช้ชุดข้อมูลรูปภาพไพ่ของแต่ละหน้า ซึ่งที่แบ่งออกเป็น 3 ส่วน:")
st.markdown("""
- **Training Set**  – ใช้สำหรับฝึกโมเดล
- **Validation Set**  – ใช้สำหรับปรับค่าพารามิเตอร์
- **Test Set**  – ใช้ตรวจสอบความสามารถของโมเดล

รูปภาพทั้งหมดถูกปรับขนาดเป็น 224x224 พิกเซล และทำ **ปรับข้อมูลให้เหมาะสม (Data Augmentation)** เช่น:
- การหมุนภาพ (Rotation)
- การขยายและย่อภาพ (Zoom & Scaling)
- การปรับความสว่างของภาพ (Brightness Adjustment)
- การแปลงเชิงเรขาคณิต (Shearing)
""")

st.subheader("2️. ทฤษฎีของอัลกอริทึมที่พัฒนา")
st.write("โมเดลใช้โครงสร้าง CNN (Convolutional Neural Network) ซึ่งเหมาะสำหรับการจำแนกภาพ")
st.markdown("""
CNN ประกอบด้วย 4 ชั้นหลัก:
1. **Convolutional Layers** – ใช้ฟิลเตอร์ขนาด 3x3 เพื่อตรวจจับรูปแบบต่างๆ บนไพ่
2. **MaxPooling Layers** – ลดขนาดของฟีเจอร์แมพ เพื่อให้โมเดลเรียนรู้เร็วขึ้น
3. **Flatten Layer** – แปลงภาพเป็นเวกเตอร์เพื่อนำไปเข้า Fully Connected Layer
4. **Fully Connected Layers** – ใช้โหนด 512 ตัวพร้อม ReLU activation ก่อนส่งเข้า Softmax เพื่อทำนายไพ่

โมเดลใช้ **Categorical Crossentropy** เป็น Loss Function และ **Adam Optimizer** สำหรับการฝึก
""")


st.subheader("3️. ขั้นตอนการพัฒนาโมเดล")
st.write("กระบวนการพัฒนาโมเดลมีดังนี้:")
st.markdown("""
1. **โหลดชุดข้อมูล** – ใช้ `ImageDataGenerator` เพื่อโหลดข้อมูลและทำ Data Augmentation
2. **สร้างโมเดล CNN** – ประกอบไปด้วย 3 ชั้น Convolutional Layers และ Fully Connected Layers
3. **คอมไพล์โมเดล** – ใช้ `Adam Optimizer` และ `Categorical Crossentropy`
4. **เทรนโมเดล** – ใช้ `model.fit()` เทรนโมเดลเป็นเวลา 30 Epochs
5. **ปรับปรุงโมเดล** – ใช้ `EarlyStopping` และ `ReduceLROnPlateau` เพื่อลด Overfitting
6. **บันทึกโมเดล** – บันทึกโมเดลในรูปแบบ `.keras` เพื่อใช้งานใน Streamlit
7. **บันทึกโมเดลลง Google Drive** - เพราะขนาดโมเดลที่ใหญ่เกินจะอัปโหลดขึ้น `github` ได้จึงใช้วิธีฝากไฟล์ไว้ใน [Google Drive](https://drive.google.com/file/d/1VZ6VCg-JqArqnZkiqCLEpAhM0JoWQNCw/view?usp=drive_link) และดึงมาใช้งานบน Streamlit แทน
""")


st.subheader("4️. สรุปผล (Conclusion)")
st.write("โมเดลสามารถทำนายไพ่ได้แม่นยำถึง **89%** โดยใช้ CNN")
