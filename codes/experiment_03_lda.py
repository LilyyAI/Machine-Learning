
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = {
    'Study_Hours': [5, 3, 6, 4, 7, 2, 5],
    'Attendance': [85, 70, 90, 65, 95, 60, 80],
    'Marks': [78, 65, 88, 72, 95, 55, 82],
    'Result': ['Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Marks']]
y = df['Result']

print("Original Dataset")
print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=1)
lda_data = lda.fit_transform(X_scaled, y)

result = pd.DataFrame(lda_data, columns=['LD1'])

print("\nLDA Transformed Dataset")
print(result.round(2))

print("\nClass Labels")
print(y.tolist())
