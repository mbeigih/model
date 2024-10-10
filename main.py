from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# بارگذاری مدل ذخیره شده
model = joblib.load('diabetes_model.pkl')

# ایجاد اپلیکیشن FastAPI
app = FastAPI()

# تعریف کلاس ورودی برای API
class DiabetesData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float

# روت برای دریافت اطلاعات و پیش‌بینی دیابت
@app.post("/predict_diabetes/")
def predict_diabetes(data: DiabetesData):
    # تبدیل داده‌های ورودی به فرمت numpy array
    input_data = np.array([[data.feature1, data.feature2, data.feature3, data.feature4, data.feature5, 
                            data.feature6, data.feature7, data.feature8, data.feature9, data.feature10]])

    # استفاده از مدل برای پیش‌بینی
    prediction = model.predict(input_data)

    # برگرداندن نتیجه
    return {"prediction": int(prediction[0])}

# برای اجرای سرور:
# uvicorn main:app --reload
