import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest

# Load the graph
G = nx.read_graphml('synthetic_graph_4.graphml')

# Extract PCA features and ground truth labels from the graph
pca_features = np.array([[G.nodes[n][f'PCA_Feature_{i+1}'] for i in range(25)] for n in G.nodes()])
ground_truth = np.array([G.nodes[n]['GT'] for n in G.nodes()])

# Initialize the Isolation Forest model
clf = IsolationForest(random_state=42, n_estimators=10, contamination=0.095,max_samples='auto')

# Fit the model to the PCA features, clf also takes the ground truth labels
clf.fit(pca_features, ground_truth)
# Predictions: -1 for outliers and 1 for inliers
is_inlier = clf.predict(pca_features)

# Find the indices (node labels) of the outliers
outlier_nodes = [n for i, n in enumerate(G.nodes()) if is_inlier[i] == -1]

# Print the results
print(f"Total nodes: {len(G.nodes())}")
print(f"Anomalous nodes detected: {len(outlier_nodes)}")
print(f"Detected anomalous node IDs: {outlier_nodes}")

# Check the unique values of ground truth labels
unique_labels = np.unique(ground_truth)
print(f"Unique labels in the ground truth: {unique_labels}")

# Make sure the target names correspond to the unique labels found in ground truth
# If there are indeed only two unique labels (0 and 1), this error shouldn't occur
from sklearn.metrics import classification_report
# Remap is_inlier to match the ground truth encoding
is_inlier_remapped = np.where(is_inlier == -1, 1, 0)

# Now use the remapped predictions for the classification report
print(classification_report(ground_truth, is_inlier_remapped, target_names=['Normal', 'Anomalous']))
