
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = {
    'Study_Hours': [1, 2, 2, 3, 6, 7, 8, 9],
    'Attendance': [40, 45, 50, 55, 80, 85, 90, 95]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

scaler = StandardScaler()
X = scaler.fit_transform(df)

model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

df['Cluster'] = model.labels_

print("\nClustered Dataset")
print(df)

print("\nCluster Centers")
print(scaler.inverse_transform(model.cluster_centers_).round(2))

print("\nInertia")
print(round(model.inertia_, 2))

plt.scatter(
    df['Study_Hours'],
    df['Attendance'],
    c=df['Cluster'],
    s=100
)

plt.xlabel('Study Hours')
plt.ylabel('Attendance')
plt.title('K-Means Clustering')
plt.show()
