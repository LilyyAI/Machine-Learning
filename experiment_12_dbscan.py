import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 15],
    'Marks': [35, 40, 45, 55, 65, 75, 85, 95, 20]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
X = scaler.fit_transform(df)

dbscan = DBSCAN(eps=0.8, min_samples=2)
df['Cluster'] = dbscan.fit_predict(X)

print("Original Dataset:")
print(df[['Study_Hours', 'Marks']])

print("\nDBSCAN Clustering Results:")
print(df)

print("\nUnique Clusters:")
print(df['Cluster'].unique())

plt.scatter(df['Study_Hours'], df['Marks'], c=df['Cluster'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("DBSCAN Clustering")
plt.show()