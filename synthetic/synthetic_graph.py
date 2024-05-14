import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Parameters
num_users = 10000  # Number of users
original_num_features = 100  # Starting with 100 features
num_pca_features = 25  # Number of PCA components to keep
percent_anomalous = 9.5
probability_of_edge_creation = 0.01  # Probability for edge creation in Erdős–Rényi model

# Set seed for reproducibility
np.random.seed(0)

# Generate random data for the original 100 features
features = np.random.rand(num_users, original_num_features)

# Perform PCA on the features
pca = PCA(n_components=num_pca_features)
pca_features = pca.fit_transform(features)

# Create a DataFrame with PCA features
df = pd.DataFrame(pca_features, columns=[f'PCA_Feature_{i+1}' for i in range(num_pca_features)])

# Add a ground truth (GT) column for anomalies
df['GT'] = np.random.choice([0, 1], size=num_users, p=[1 - percent_anomalous/100, percent_anomalous/100])

# Create an Erdős–Rényi graph
G = nx.gnp_random_graph(n=num_users, p=probability_of_edge_creation)

# Assign PCA features and class to each node in the graph
for i, row in df.iterrows():
    G.nodes[i].update(row.to_dict())
    if i%1000 == 0:
        print(f"Assigned features to {i} nodes")
        
print("Assigned features to all nodes")

#print the stats of the graph

print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
print("Number of normal nodes: ", sum(df['GT']))

'''
# Plot the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)
plt.title('Synthetic Graph with PCA Features')
plt.savefig("synthetic_graph_4.png")
plt.close()  # Close the figure to prevent it from displaying in the notebook/output
'''
# Save the graph as GraphML
nx.write_graphml(G, "synthetic_graph_5.graphml")

