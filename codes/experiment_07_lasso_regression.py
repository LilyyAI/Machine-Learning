
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Attendance': [50, 55, 60, 65, 70, 80, 90, 95],
    'Marks': [35, 40, 48, 55, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Marks']

print("Original Dataset")
print(df)

model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=10000)
)

model.fit(X, y)

predictions = model.predict(X)

print("\nActual and Predicted Marks")
result = pd.DataFrame({
    'Actual': y,
    'Predicted': predictions.round(2)
})
print(result)

print("\nMean Squared Error")
print(round(mean_squared_error(y, predictions), 2))

print("\nR2 Score")
print(round(r2_score(y, predictions), 2))

lasso = model.named_steps['lasso']
print("\nLasso Coefficients")
print(lasso.coef_.round(3))

new_student = pd.DataFrame({
    'Study_Hours': [5],
    'Attendance': [80]
})

prediction = model.predict(new_student)

print("\nPredicted Marks")
print(round(prediction[0], 2))
