# Import کتابخانه‌ها
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# بارگذاری دیتاست دیابت
diabetes_data = load_diabetes()

# جدا کردن ویژگی‌ها و برچسب‌ها (Features و Labels)
X = diabetes_data.data
y = diabetes_data.target

# نرمال‌سازی برچسب‌ها به اعداد باینری (اختیاری: مثلا بیماری یا سالم)
# به دلیل اینکه target مقادیر پیوسته دارد، باید آن را به طبقه‌بندی تبدیل کنیم.
y = (y > y.mean()).astype(int)  # اگر مقدار بالاتر از میانگین باشد 1، در غیر این صورت 0

# تقسیم داده‌ها به دو مجموعه آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# انتخاب و آموزش مدل (در اینجا Logistic Regression)
model = LogisticRegression(max_iter=10000)
# یا به جای LogisticRegression می‌توانید RandomForestClassifier را امتحان کنید:
# model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# پیش‌بینی روی مجموعه آزمایشی
y_pred = model.predict(X_test)

# ارزیابی دقت مدل
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.4f}")

# ذخیره مدل با استفاده از joblib
joblib.dump(model, 'diabetes_model.pkl')
print("Model saved as 'diabetes_model.pkl'")
