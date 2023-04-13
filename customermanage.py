import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import cdist

# Load customer data from a CSV file
data = pd.read_csv('customer_data.csv')

# Define the features you want to use for clustering and classification
features = ['type','location','rooms','employees','revenue']

# Convert categorical features to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['type', 'location'])

# Normalize the features to ensure they have the same scale
data[features] = (data[features] - data[features].mean()) / data[features].std()

# Define the parameter grid for K-Means clustering
kmeans_params = {'n_clusters': [2, 3, 4, 5], 'random_state': [0]}

# Perform grid search to find the best K-Means parameters
kmeans = KMeans()
kmeans_gs = GridSearchCV(kmeans, kmeans_params)
kmeans_gs.fit(data[features])
clusters = kmeans_gs.predict(data[features])

# Add the cluster assignments to the customer data
data['cluster'] = clusters

# Define the parameter grid for Decision Tree classifier
dt_params = {'max_depth': [2, 3, 4, 5], 'random_state': [0]}

# Perform grid search to find the best Decision Tree parameters
dt = DecisionTreeClassifier()
dt_gs = GridSearchCV(dt, dt_params)
dt_gs.fit(data[features], clusters)

# Define the parameter grid for Logistic Regression classifier
lr_params = {'C': [0.1, 1, 10], 'random_state': [0]}

# Perform grid search to find the best Logistic Regression parameters
lr = LogisticRegression()
lr_gs = GridSearchCV(lr, lr_params)
lr_gs.fit(data[features], clusters)

# Define the parameter grid for Random Forest classifier
rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [2, 3, 4, 5], 'random_state': [0]}

# Perform grid search to find the best Random Forest parameters
rf = RandomForestClassifier()
rf_gs = GridSearchCV(rf, rf_params)
rf_gs.fit(data[features], clusters)

# Use the classifiers to predict the cluster assignments of new customers
new_customer = np.array([[101, "New Hotel", "Hotel", "B", 50, 20, 5000000]])
new_customer = pd.DataFrame(new_customer, columns=['id', 'name', 'type', 'location', 'rooms', 'employees', 'revenue'])
new_customer = pd.get_dummies(new_customer, columns=['type', 'location'])
new_customer[features] = (new_customer[features] - data[features].mean()) / data[features].std()

dt_prediction = dt_gs.predict(new_customer[features])
lr_prediction = lr_gs.predict(new_customer[features])
rf_prediction = rf_gs.predict(new_customer[features])

print("Decision Tree prediction: ", dt_prediction)
print("Logistic Regression prediction: ", lr_prediction)
print("Random Forest prediction: ", rf_prediction)

# Save the updated customer data to a CSV file
data.to_csv('customer_data_clustered.csv', index=False)
