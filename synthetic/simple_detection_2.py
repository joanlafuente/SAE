from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load the graph
G = nx.read_graphml('synthetic_graph_4.graphml')

# Extract PCA features and ground truth labels from the graph
pca_features = np.array([[G.nodes[n][f'PCA_Feature_{i+1}'] for i in range(25)] for n in G.nodes()])
ground_truth = np.array([G.nodes[n]['GT'] for n in G.nodes()])


# Split data into train and test to evaluate model
X_train, X_test, y_train, y_test = train_test_split(pca_features, ground_truth, test_size=0.3, random_state=42)

# Initialize the model with new parameters
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Make predictions on the test set
is_inlier_test = clf.predict(X_test)
is_inlier_test_remapped = np.where(is_inlier_test == -1, 1, 0)

# Evaluate the model
print(classification_report(y_test, is_inlier_test_remapped, target_names=['Normal', 'Anomalous']))
print("ROC AUC score:", roc_auc_score(y_test, clf.decision_function(X_test)))
