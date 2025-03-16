import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title= "Neural Network Demo",
    page_icon="♠️",
)

st.title("Cards classification")
st.write("อัปโหลดรูปภาพไพ่")

# โหลดโมเดลที่เทรนไว้
model = tf.keras.models.load_model("Neural_Network/card_classifier.keras")

# รายชื่อคลาส (53 ไพ่)
train_dir = "Neural_Network/train"
class_names = sorted(os.listdir(train_dir)) 

# อัปโหลดไฟล์รูป
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)
    
    # แปลงภาพให้เข้ากับโมเดล
    image = image.convert("RGB")  # บังคับให้เป็น RGB
    image = image.resize((224, 224))  # ปรับขนาด
    image = np.array(image) / 255.0  # ทำ Normalization
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้เข้ากับโมเดล
    
    # ทำการพยากรณ์
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]
    confidence = np.max(prediction) * 100
    
    # แสดงผลลัพธ์
    st.subheader(f"ไพ่นี้คือ: {predicted_class}")
    
