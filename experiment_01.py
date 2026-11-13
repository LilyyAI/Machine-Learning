import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = {
    'Study_Hours': [5, 3, 6, 4, 7, 2, 5],
    'Attendance': [85, 70, 90, None, 95, 60, 80],
    'Marks': [78, 55, 88, 65, 92, 45, 75],
    'Result': ['Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)

df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())

encoder = LabelEncoder()
df['Result'] = encoder.fit_transform(df['Result'])

scaler = StandardScaler()
df[['Study_Hours', 'Attendance', 'Marks']] = scaler.fit_transform(
    df[['Study_Hours', 'Attendance', 'Marks']]
)

print(df)