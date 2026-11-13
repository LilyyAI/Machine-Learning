import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9],
    'Marks': [35, 40, 45, 55, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
X = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print("Original Dataset:")
print(df[['Study_Hours', 'Marks']])

print("\nCluster Results:")
print(df)

print("\nCluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))

plt.scatter(df['Study_Hours'], df['Marks'], c=df['Cluster'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("K-Means Clustering")
plt.show()