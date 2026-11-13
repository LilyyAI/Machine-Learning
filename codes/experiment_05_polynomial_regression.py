
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [35, 40, 48, 58, 70, 78, 88, 96]
}

df = pd.DataFrame(data)

X = df[['Study_Hours']]
y = df['Marks']

model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X, y)

predictions = model.predict(X)

print("Original Dataset")
print(df)

print("\nPredicted Marks")
print(predictions.round(2))

new_data = pd.DataFrame({'Study_Hours': [5.5]})
prediction = model.predict(new_data)

print("\nPredicted Marks for 5.5 Study Hours")
print(round(prediction[0], 2))

plt.scatter(X, y, label='Actual Data')
plt.plot(X, predictions, label='Polynomial Regression')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
