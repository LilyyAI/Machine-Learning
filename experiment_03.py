import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = {
    'Study_Hours': [5, 3, 6, 4, 7, 2, 5, 8],
    'Attendance': [85, 60, 90, 70, 95, 50, 80, 98],
    'Marks': [78, 45, 88, 55, 92, 35, 75, 96],
    'Result': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

encoder = LabelEncoder()
df['Result'] = encoder.fit_transform(df['Result'])

X = df[['Study_Hours', 'Attendance', 'Marks']]
y = df['Result']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=1)
result = lda.fit_transform(X_scaled, y)

lda_df = pd.DataFrame(result, columns=['LD1'])

print("Original Dataset:")
print(df)

print("\nLDA Result:")
print(lda_df)

print("\nClass Labels:")
print(y)