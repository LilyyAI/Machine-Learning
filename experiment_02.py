import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    'Study_Hours': [5, 3, 6, 4, 7, 2, 5],
    'Attendance': [85, 70, 90, 65, 95, 60, 80],
    'Marks': [78, 55, 88, 65, 92, 45, 75]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(result, columns=['PC1', 'PC2'])

print("Original Dataset:")
print(df)

print("\nPCA Result:")
print(pca_df)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)