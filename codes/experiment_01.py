
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = {
    'Study_Hours': [5, 3, 6, 4, 7, 2, 5],
    'Attendance': [85, 70, 90, None, 95, 60, 80],
    'Marks': [78, 65, 88, 72, None, 55, 82],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

print("\nMissing Values")
print(df.isnull().sum())

df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

scaler = StandardScaler()
cols = ['Study_Hours', 'Attendance', 'Marks']
df[cols] = scaler.fit_transform(df[cols])

print("\nFinal Processed Dataset")
print(df.round(2))
