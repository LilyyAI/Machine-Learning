import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 6, 8],
    'Attendance': [50, 60, 65, 70, 80, 85, 90, 95, 55, 68, 82, 92],
    'Marks': [35, 40, 45, 55, 65, 75, 85, 95, 38, 48, 70, 90],
    'Result': ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass',
               'Pass', 'Pass', 'Fail', 'Fail', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Marks']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))