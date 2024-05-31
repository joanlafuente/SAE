from scipy.io import loadmat
import pickle
from collections import Counter
import numpy as np
import networkx as nx
from pyvis.network import Network 
import os
from scipy.sparse import csr_matrix
from collections import deque
from pyvis.network import Network



def Amazon():
    data_file = loadmat('./Amazon.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    # Convert the data_file into a nx graph but only taking the first 1000 nodes and their edges, with node ids starting from 0
    feat_data = feat_data[:1500,:]
    labels = labels[:500]
    # Create the graph
    print("Creating the graph")
    G = nx.Graph()
    for i in range(feat_data.shape[0]):
        G.add_node(i, title=str(i))
        for j in range(feat_data.shape[1]):
            if feat_data[i,j] == 1: 
                G.add_edge(i,j)


    print("Graph created")
    # Save the graph in pyvis format
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)
    print("Adding nodes and edges")
    graph.from_nx(G)
    print("Nodes and edges added")
    graph.show_buttons(filter_=['physics'])
    graph.toggle_physics(False)
    graph.write_html('./static/Amazon.html')


def bfs_collect_nodes(graph, start_node, max_nodes):
    visited = set()
    queue = deque([start_node])
    bfs_nodes = []

    while queue and len(bfs_nodes) < max_nodes:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            bfs_nodes.append(node)
            # Get the neighbors from the graph
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    return bfs_nodes

def bfs_collect_nodes_with_predictions(graph, start_node, max_nodes):
    visited = set()
    queue = deque([start_node])
    bfs_nodes = []

    while queue and len(bfs_nodes) < max_nodes:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            bfs_nodes.append(node)
            # Get the neighbors from the graph
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    return bfs_nodes

def Yelp():
    # Load the data
    data_file = loadmat('./YelpChi.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    # Convert the feature data to a sparse matrix
    feat_data_sparse = csr_matrix(feat_data)

    # Create the full graph from the feature matrix
    G = nx.Graph()
    num_nodes = feat_data_sparse.shape[0]

    for i in range(num_nodes):
        G.add_node(i)
        for j in feat_data_sparse[i].indices:
            G.add_edge(i, j)

    # Perform BFS to collect the first 5000 nodes
    bfs_nodes = bfs_collect_nodes(G, start_node=0, max_nodes=1500)

    # Create the subgraph with the collected BFS nodes
    subgraph = G.subgraph(bfs_nodes).copy()

    print(f"Subgraph created with {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges")

    # Save the subgraph in pyvis format
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)
    print("Adding nodes and edges")
    
    # Explicitly convert node IDs to integers when adding to pyvis
    for node, data in subgraph.nodes(data=True):
        graph.add_node(int(node), **data)
    for u, v, data in subgraph.edges(data=True):
        graph.add_edge(int(u), int(v), **data)
    
    print("Nodes and edges added")
    graph.show_buttons(filter_=['physics'])
    graph.toggle_physics(False)
    graph.write_html('./static/Yelp.html')
    print("HTML file created")

def YelpSupervised():
    # Load the data
    data_file = loadmat('./YelpChi.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    # Convert the feature data to a sparse matrix
    feat_data_sparse = csr_matrix(feat_data)

    # Create the full graph from the feature matrix
    G = nx.Graph()
    num_nodes = feat_data_sparse.shape[0]

    for i in range(num_nodes):
        G.add_node(i)
        for j in feat_data_sparse[i].indices:
            G.add_edge(i, j)

    # Load the anomaly predictions
    predictions = np.load('./static/predictions/YelpSupervised_predictions.npy')

    # Convert predictions to float type
    predictions = predictions.astype(float)

    # Assign predictions to each node
    for i in range(num_nodes):
        G.nodes[i]['prediction'] = predictions[i]

    # Perform BFS to collect the first 5000 nodes
    bfs_nodes = bfs_collect_nodes_with_predictions(G, start_node=7000, max_nodes=1500)

    # Create the subgraph with the collected BFS nodes
    subgraph = G.subgraph(bfs_nodes).copy()

    print(f"Subgraph created with {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges")

    # Save the subgraph in pyvis format
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)
    print("Adding nodes and edges")

    # Explicitly convert node IDs to integers when adding to pyvis and set color based on anomaly probability
    for node, data in subgraph.nodes(data=True):
        color = 'red' if data['prediction'] > 0.5 else '#97c2fc'  # Light blue color
        title = f"Node {node} - Anomaly: {data['prediction']:.2f}"
        graph.add_node(int(node), color=color, title=title, **data)
    for u, v, data in subgraph.edges(data=True):
        graph.add_edge(int(u), int(v), **data)
    
    print("Nodes and edges added")
    graph.show_buttons(filter_=['physics'])
    graph.toggle_physics(False)
    graph.write_html('./static/YelpSupervised.html')
    print("HTML file created")

def AmazonSupervised():
    data_file = loadmat('./Amazon.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    # Convert the data_file into a nx graph but only taking the first 1000 nodes and their edges, with node ids starting from 0
    feat_data = feat_data[:1500,:]
    labels = labels[:500]
    # Create the graph
    print("Creating the graph")
    G = nx.Graph()
    feat_data_sparse = csr_matrix(feat_data)
    for i in range(feat_data.shape[0]):
        G.add_node(i)
        for j in range(feat_data.shape[1]):
            if feat_data[i,j] == 1: 
                G.add_edge(i,j)


    # Load the anomaly predictions
    predictions = np.load('./static/predictions/Amazon_Supervised_predictions.npy')

    # Convert predictions to float type
    predictions = predictions.astype(float)
    num_nodes = feat_data_sparse.shape[0]
    # Assign predictions to each node
    for i in range(num_nodes):
        G.nodes[i]['prediction'] = predictions[i]

    print("Graph created")
    # Save the graph in pyvis format
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)
    print("Adding nodes and edges")
    # Explicitly convert node IDs to integers when adding to pyvis and set color based on anomaly probability
    for node, data in G.nodes(data=True):
        color = 'red' if data['prediction'] > 0.5 else '#97c2fc'  # Light blue color
        title = f"Node {node} - Anomaly: {data['prediction']:.2f}"
        graph.add_node(int(node), color=color, title=title, **data)
    for u, v, data in G.edges(data=True):
        graph.add_edge(int(u), int(v), **data)

    print("Adding nodes and edges")
    graph.from_nx(G)
    print("Nodes and edges added")
    graph.show_buttons(filter_=['physics'])
    graph.toggle_physics(False)
    graph.write_html('./static/AmazonSupervised.html')


def YelpAutoencoders():
    # Load the data
    data_file = loadmat('./YelpChi.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    # Convert the feature data to a sparse matrix
    feat_data_sparse = csr_matrix(feat_data)

    # Create the full graph from the feature matrix
    G = nx.Graph()
    num_nodes = feat_data_sparse.shape[0]

    for i in range(num_nodes):
        G.add_node(i)
        for j in feat_data_sparse[i].indices:
            G.add_edge(i, j)

    # Load the anomaly predictions
    predictions = np.load('./static/predictions/YelpAutoencoders.npy')

    # Convert predictions to float type
    predictions = predictions.astype(float)

    # Assign predictions to each node
    for i in range(num_nodes):
        G.nodes[i]['prediction'] = predictions[i]

    # Perform BFS to collect the first 5000 nodes
    bfs_nodes = bfs_collect_nodes_with_predictions(G, start_node=7000, max_nodes=1500)

    # Create the subgraph with the collected BFS nodes
    subgraph = G.subgraph(bfs_nodes).copy()

    print(f"Subgraph created with {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges")

    # Save the subgraph in pyvis format
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)
    print("Adding nodes and edges")

    # Explicitly convert node IDs to integers when adding to pyvis and set color based on anomaly probability
    for node, data in subgraph.nodes(data=True):
        color = 'red' if data['prediction'] > 0.5 else '#97c2fc'  # Light blue color
        title = f"Node {node} - Anomaly: {data['prediction']:.2f}"
        graph.add_node(int(node), color=color, title=title, **data)
    for u, v, data in subgraph.edges(data=True):
        graph.add_edge(int(u), int(v), **data)
    
    print("Nodes and edges added")
    graph.show_buttons(filter_=['physics'])
    graph.toggle_physics(False)
    graph.write_html('./static/YelpAutoencoders.html')
    print("HTML file created")

def AmazonAutoencoders():
    data_file = loadmat('./Amazon.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    # Convert the data_file into a nx graph but only taking the first 1000 nodes and their edges, with node ids starting from 0
    feat_data = feat_data[:1500,:]
    labels = labels[:500]
    # Create the graph
    print("Creating the graph")
    G = nx.Graph()
    feat_data_sparse = csr_matrix(feat_data)
    for i in range(feat_data.shape[0]):
        G.add_node(i)
        for j in range(feat_data.shape[1]):
            if feat_data[i,j] == 1: 
                G.add_edge(i,j)


    # Load the anomaly predictions
    predictions = np.load('./static/predictions/Amazon_Autoencoder.npy')

    # Convert predictions to float type
    predictions = predictions.astype(float)
    num_nodes = feat_data_sparse.shape[0]
    # Assign predictions to each node
    for i in range(num_nodes):
        G.nodes[i]['prediction'] = predictions[i]

    print("Graph created")
    # Save the graph in pyvis format
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)
    print("Adding nodes and edges")
    # Explicitly convert node IDs to integers when adding to pyvis and set color based on anomaly probability
    for node, data in G.nodes(data=True):
        color = 'red' if data['prediction'] > 0.5 else '#97c2fc'  # Light blue color
        title = f"Node {node} - Anomaly: {data['prediction']:.2f}"
        graph.add_node(int(node), color=color, title=title, **data)
    for u, v, data in G.edges(data=True):
        graph.add_edge(int(u), int(v), **data)

    print("Adding nodes and edges")
    graph.from_nx(G)
    print("Nodes and edges added")
    graph.show_buttons(filter_=['physics'])
    graph.toggle_physics(False)
    graph.write_html('./static/AmazonAutoencoders.html')

def AmazonSelfSupervised():
    data_file = loadmat('./Amazon.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    # Convert the data_file into a nx graph but only taking the first 1000 nodes and their edges, with node ids starting from 0
    feat_data = feat_data[:1500,:]
    labels = labels[:500]
    # Create the graph
    print("Creating the graph")
    G = nx.Graph()
    feat_data_sparse = csr_matrix(feat_data)
    for i in range(feat_data.shape[0]):
        G.add_node(i)
        for j in range(feat_data.shape[1]):
            if feat_data[i,j] == 1: 
                G.add_edge(i,j)


    # Load the anomaly predictions
    predictions = np.load('./static/predictions/Amazon_SelfSupervisedC.npy')

    # Convert predictions to float type
    predictions = predictions.astype(float)
    num_nodes = feat_data_sparse.shape[0]
    # Assign predictions to each node
    for i in range(num_nodes):
        G.nodes[i]['prediction'] = predictions[i]

    print("Graph created")
    # Save the graph in pyvis format
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)
    print("Adding nodes and edges")
    # Explicitly convert node IDs to integers when adding to pyvis and set color based on anomaly probability
    for node, data in G.nodes(data=True):
        color = 'red' if data['prediction'] > 0.5 else '#97c2fc'  # Light blue color
        title = f"Node {node} - Anomaly: {data['prediction']:.2f}"
        graph.add_node(int(node), color=color, title=title, **data)
    for u, v, data in G.edges(data=True):
        graph.add_edge(int(u), int(v), **data)

    print("Adding nodes and edges")
    graph.from_nx(G)
    print("Nodes and edges added")
    graph.show_buttons(filter_=['physics'])
    graph.toggle_physics(False)
    graph.write_html('./static/AmazonSelfSupervised.html')

#Yelp()
#Amazon()
#YelpSupervised()
#AmazonSupervised()
#YelpAutoencoders()
#AmazonAutoencoders()
AmazonSelfSupervised()