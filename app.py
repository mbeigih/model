from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# بارگذاری مدل
model = joblib.load("diabetes_model.pkl")

@app.post("/predict/")
def predict(data: dict):
    X = np.array([data['features']])
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}

# برای اجرای سرور:
# uvicorn app:app --reload
