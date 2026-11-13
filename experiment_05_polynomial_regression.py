import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [35, 40, 42, 50, 60, 72, 78, 85, 92, 98]
}

df = pd.DataFrame(data)

X = df[['Study_Hours']]
y = df['Marks']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

print("Original Dataset:")
print(df)

print("\nPredicted Marks:")
print(y_pred)

print("\nMean Squared Error:")
print(mean_squared_error(y, y_pred))

print("\nR2 Score:")
print(r2_score(y, y_pred))

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Polynomial Regression")
plt.show()