
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    'Study_Hours': [5, 3, 6, 4, 7, 2, 5],
    'Attendance': [85, 70, 90, 65, 95, 60, 80],
    'Marks': [78, 65, 88, 72, 95, 55, 82]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

result = pd.DataFrame(
    pca_data,
    columns=['PC1', 'PC2']
)

print("\nPCA Transformed Dataset")
print(result.round(2))

print("\nExplained Variance Ratio")
print(pca.explained_variance_ratio_.round(3))

print("\nTotal Explained Variance")
print(round(pca.explained_variance_ratio_.sum(), 3))
