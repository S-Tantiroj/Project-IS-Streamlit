import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("Salary Data.csv")

# ลบแถวที่มีค่าหายไป
df_cleaned = df.dropna()

# แปลงข้อมูลประเภทข้อความเป็นตัวเลข
label_encoders = {}
categorical_columns = ["Gender", "Education Level", "Job Title"]

for col in categorical_columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# แยก Features และ Target
X = df_cleaned.drop(columns=["Salary"])     #Features
y = df_cleaned["Salary"]                    #Target

# แบ่งข้อมูลเป็น Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับขนาดข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้างโมเดล
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, early_stopping_rounds=5, random_state=42)

# ฝึกโมเดล
rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

# ทำนายผล
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_gb = gb_model.predict(X_test_scaled)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# คำนวณค่าประสิทธิภาพของโมเดล
def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred)
    }

# สรุปผล
model_results = {
    "Random Forest": evaluate_model(y_test, y_pred_rf),
    "Gradient Boosting": evaluate_model(y_test, y_pred_gb),
    "XGBoost": evaluate_model(y_test, y_pred_xgb)
}

# แสดงผลลัพธ์
df_results = pd.DataFrame(model_results).T
print(df_results)

# คัดเลือกโมเดลที่ดีที่สุด
best_model_name = max(model_results, key=lambda x: model_results[x]["R²"])
best_model = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model
}[best_model_name]

# บันทึกโมเดลที่ดีที่สุด
joblib.dump(best_model, "best_model.pkl")

# บันทึกตัวเข้ารหัสและ Scaler
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"Best model: {best_model_name} -> best_model.pkl")