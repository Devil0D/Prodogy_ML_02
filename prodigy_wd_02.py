import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the customer purchase history dataset
try:
    data = pd.read_csv('./dataset/Mall_customers.csv')  # Adjust the path as necessary
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Display the first few rows of the data to understand its structure
print(data.head())

# 1. Select features relevant for clustering (excluding 'CustomerID' and 'Gender' for clustering)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# 2. Extract features for clustering
X = data[features]

# 3. Preprocess the data
# Standardize numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Encode categorical 'Gender' column
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# 5. Choose the number of clusters (k) for K-Means
# Let's use the Elbow Method to determine the optimal k
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve to visualize the optimal k
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.savefig('elbow_method_plot.png')  # Save the Elbow plot as a file
print("Elbow plot saved as 'elbow_method_plot.png'")

# Based on the Elbow curve, select an optimal k (e.g., k=3 is often a good starting point)
optimal_k = 3  # Modify based on the elbow plot

# 6. Apply K-Means with the selected number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 7. Check the clusters
print(data[['CustomerID', 'Gender', 'Cluster']].head())

# 8. Evaluate the clustering (optional)
silhouette_avg = silhouette_score(X_scaled, data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.2f}")

# 9. Visualize the clusters (optional)
# Plot based on the first two features (Age vs Spending Score)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segments (Clusters)')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.colorbar(label='Cluster')
plt.savefig('clusters_plot.png')  # Save the customer clusters plot
print("Clusters plot saved as 'clusters_plot.png'")

# 10. Save the clustered data to a new file (optional)
data.to_csv('clustered_customers.csv', index=False)  # Replaces the file each time
print("Clustered data saved to 'clustered_customers.csv'")
