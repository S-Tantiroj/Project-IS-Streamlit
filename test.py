import pickle
#เป็นไฟล์ที่ใช้ทดลองทดสอบการทำงานในแต่ละส่วนเท่านั้น ไม่ได้เชื่อโยงไปยังการทำงานหลัก

# 🔹 โหลดโมเดลที่บันทึกไว้
with open("Machine_learning/model.pkl", "rb") as file:
    data = pickle.load(file)

model = data["model"]  # ดึงโมเดลออกมา
features = data["features"]  # ดึงฟีเจอร์ที่ใช้ตอนเทรน

# 🔹 ตรวจสอบฟีเจอร์
if features:
    print("✅ Features used in model:", features)
else:
    print("⚠️ No feature names found in model.")
