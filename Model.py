from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# بارگذاری دیتاست دیابت
data = load_diabetes()
X = data.data
y = (data.target > 150).astype(int)  # تبدیل هدف به دسته‌های باینری

# تقسیم داده‌ها به داده‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ساخت مدل
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ارزیابی مدل
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ذخیره مدل
joblib.dump(model, "diabetes_model.pkl")
