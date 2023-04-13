import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('k-mean-data.csv')

# Perform K-Means clustering with optimal number of clusters
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['age', 'income']])
    inertias.append(kmeans.inertia_)
diff = pd.Series(inertias[:-1]).diff() / inertias[:-1]
k_opt = diff.idxmin() + 2
kmeans = KMeans(n_clusters=k_opt, random_state=42)
kmeans.fit(data[['age', 'income']])
data['cluster'] = kmeans.predict(data[['age', 'income']])

# Plot clusters
colors = {'M': 'blue', 'F': 'red'}
fig, ax = plt.subplots()
for gender in data['gender'].unique():
    cluster_data = data[data['gender'] == gender]
    ax.scatter(cluster_data['age'], cluster_data['income'], c=colors[gender], label=gender, alpha=0.5)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='black', s=200, linewidth=3)
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.legend()
plt.show()
