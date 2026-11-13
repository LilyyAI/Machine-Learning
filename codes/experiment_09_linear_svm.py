
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Study_Hours': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    'Attendance': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Result': ['Fail', 'Fail', 'Fail', 'Fail', 'Fail', 'Pass',
               'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Result']

print("Original Dataset")
print(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1.0)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nActual and Predicted Results")
print(pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions
}))

print("\nAccuracy")
print(accuracy_score(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions, zero_division=0))

new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [85]
})

print("\nNew Student Prediction")
print(model.predict(new_student)[0])
