
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

data = {
    'Study_Hours': [1, 2, 2, 3, 6, 7, 8, 9, 15],
    'Attendance': [40, 45, 50, 55, 80, 85, 90, 95, 20]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

scaler = StandardScaler()
X = scaler.fit_transform(df)

model = DBSCAN(eps=0.8, min_samples=2)
df['Cluster'] = model.fit_predict(X)

print("\nDBSCAN Clustered Dataset")
print(df)

print("\nUnique Clusters")
print(sorted(df['Cluster'].unique()))

print("\nNumber of Clusters")
print(len(set(df['Cluster'])) - (1 if -1 in df['Cluster'].values else 0))

print("\nNoise Points")
print(df[df['Cluster'] == -1])

plt.scatter(
    df['Study_Hours'],
    df['Attendance'],
    c=df['Cluster'],
    s=100
)

plt.xlabel('Study Hours')
plt.ylabel('Attendance')
plt.title('DBSCAN Clustering')
plt.show()
