import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 3, 6],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95, 68, 82],
    'Marks': [45, 50, 55, 65, 70, 78, 85, 95, 52, 75]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(y_pred)

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))