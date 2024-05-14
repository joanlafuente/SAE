from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, auc
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

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Re-train the Isolation Forest on the balanced dataset
clf_balanced = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf_balanced.fit(X_resampled)

# Predict and evaluate again
is_inlier_test_balanced = clf_balanced.predict(X_test)
is_inlier_test_balanced_remapped = np.where(is_inlier_test_balanced == -1, 1, 0)

# Classification report
print(classification_report(y_test, is_inlier_test_balanced_remapped, target_names=['Normal', 'Anomalous']))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, clf_balanced.decision_function(X_test))
auc_score = auc(recall, precision)
roc_auc_score = roc_auc_score(y_test, clf_balanced.decision_function(X_test))
print("Precision-Recall AUC:", auc_score)
print("ROC AUC score:", roc_auc_score)
