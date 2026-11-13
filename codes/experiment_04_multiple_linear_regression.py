
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8],
    'Attendance': [60, 65, 70, 75, 80, 90, 95],
    'Assignment_Marks': [50, 55, 60, 65, 75, 85, 90],
    'Exam_Marks': [52, 58, 65, 70, 78, 88, 95]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Assignment_Marks']]
y = df['Exam_Marks']

print("Original Dataset")
print(df)

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("\nIntercept")
print(round(model.intercept_, 2))

print("\nCoefficients")
print(model.coef_)

print("\nActual and Predicted Marks")
result = pd.DataFrame({
    'Actual_Marks': y,
    'Predicted_Marks': predictions.round(2)
})

print(result)

new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [85],
    'Assignment_Marks': [80]
})

prediction = model.predict(new_student)

print("\nPredicted Marks for New Student")
print(round(prediction[0], 2))
