import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv, GAT
import torch.nn.functional as F
from scipy.io import loadmat
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import random

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import copy
import os
import yaml
import sys

from utils import *
from models import GCN, Simpler_GCN, Simpler_GCN_Conv, GCN_Att, Simpler_GCN2, GCN_Att_Drop_Multihead, GCN_Att_Not_res


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Open a yaml file with the parameters
with open(f'./Setups/SelfSupervisedContrastive/{name_yaml}.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

if params["data"] == "amz":
    run_path = f"./Runs/SelfSupervisedContrastive/Amazon/{name_yaml}"
    # Creating a folder for the run files
    if not os.path.exists(f'{run_path}'):
        os.makedirs(f'{run_path}')
        os.makedirs(f'{run_path}/Weights')
        os.makedirs(f'{run_path}/Plots')
        os.makedirs(f'{run_path}/Report')
        os.makedirs(f'{run_path}/Pickles')

    # Loading data
    data_file = loadmat('./Data/Amazon.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    train_mask = torch.zeros(11944, dtype=torch.bool)
    val_mask = torch.zeros(11944, dtype=torch.bool)
    test_mask = torch.zeros(11944, dtype=torch.bool)
    train_mask_contrastive = torch.zeros(11944, dtype=torch.bool)

    nodes = list(range(3305, 11944))
    train_nodes, test_val_nodes = train_test_split(nodes, train_size=0.6, stratify=labels[nodes], random_state=0)
    val_nodes, test_nodes = train_test_split(test_val_nodes, train_size=0.5, stratify=labels[test_val_nodes], random_state=0)
    train_nodes_contrastive = train_nodes + list(range(0, 3305))

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True
    train_mask_contrastive[train_nodes_contrastive] = True


    with open('./Data/amz_upu_adjlists.pickle', 'rb') as file:
        upu = pickle.load(file)

    with open('./Data/amz_usu_adjlists.pickle', 'rb') as file:
        usu = pickle.load(file)

    with open('./Data/amz_uvu_adjlists.pickle', 'rb') as file:
        uvu = pickle.load(file)

    edges_list_p = []
    for i in range(len(upu)):
        edges_list_p.extend([(i, node) for node in upu[i]])
    edges_list_p = np.array(edges_list_p)
    edges_list_p = edges_list_p.transpose()

    edges_list_s = []
    for i in range(len(upu)):
        edges_list_s.extend([(i, node) for node in usu[i]])
    edges_list_s = np.array(edges_list_s)
    edges_list_s = edges_list_s.transpose()

    edges_list_v = []
    for i in range(len(upu)):
        edges_list_v.extend([(i, node) for node in uvu[i]])
    edges_list_v = np.array(edges_list_v)
    edges_list_v = edges_list_v.transpose()

    # Creating graph
    graph = Data(x=torch.tensor(feat_data).float(), 
                edge_index_v=torch.tensor(edges_list_v), 
                edge_index_p=torch.tensor(edges_list_p),
                edge_index_s=torch.tensor(edges_list_s),
                y=torch.tensor(labels).type(torch.int64),
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                train_mask_contrastive=train_mask_contrastive)

elif params["data"] == "yelp":
    run_path = f"./Runs/SelfSupervisedContrastive/Yelp/{name_yaml}"
    # Creating a folder for the run files
    if not os.path.exists(f'{run_path}'):
        os.makedirs(f'{run_path}')
        os.makedirs(f'{run_path}/Weights')
        os.makedirs(f'{run_path}/Plots')
        os.makedirs(f'{run_path}/Report')
        os.makedirs(f'{run_path}/Pickles')
    # Loading data
    data_file = loadmat('./Data/YelpChi.mat')
    labels = data_file['label'].flatten()
    feat_data = data_file['features'].todense().A

    num_nodes = feat_data.shape[0]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask_contrastive = torch.zeros(num_nodes, dtype=torch.bool)

    nodes = np.arange(num_nodes)
    train_nodes, test_val_nodes = train_test_split(nodes, train_size=0.7, stratify=labels, random_state=0)
    val_nodes, test_nodes = train_test_split(test_val_nodes, train_size=0.5, stratify=labels[test_val_nodes], random_state=0)
    train_nodes_contrastive = train_nodes 

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True
    train_mask_contrastive[train_nodes_contrastive] = True


    with open('./Data/yelp_rtr_adjlists.pickle', 'rb') as file:
        upu = pickle.load(file)

    with open('./Data/yelp_rsr_adjlists.pickle', 'rb') as file:
        usu = pickle.load(file)

    with open('./Data/yelp_rur_adjlists.pickle', 'rb') as file:
        uvu = pickle.load(file)

    edges_list_p = []
    for i in range(len(upu)):
        edges_list_p.extend([(i, node) for node in upu[i]])
    edges_list_p = np.array(edges_list_p)
    edges_list_p = edges_list_p.transpose()

    edges_list_s = []
    for i in range(len(upu)):
        edges_list_s.extend([(i, node) for node in usu[i]])
    edges_list_s = np.array(edges_list_s)
    edges_list_s = edges_list_s.transpose()

    edges_list_v = []
    for i in range(len(upu)):
        edges_list_v.extend([(i, node) for node in uvu[i]])
    edges_list_v = np.array(edges_list_v)
    edges_list_v = edges_list_v.transpose()

    # Creating graph
    graph = Data(x=torch.tensor(feat_data).float(), 
                edge_index_v=torch.tensor(edges_list_v), 
                edge_index_p=torch.tensor(edges_list_p),
                edge_index_s=torch.tensor(edges_list_s),
                y=torch.tensor(labels).type(torch.int64),
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                train_mask_contrastive=train_mask_contrastive)


if params["model_name"] == 'Simpler_GCN':
    model = Simpler_GCN(**params['model'])
elif params["model_name"] == 'Simpler_GCN_Conv':
    model = Simpler_GCN_Conv(**params['model'])
elif params["model_name"] == 'Simpler_GCN2':
    model = Simpler_GCN2(**params['model'])
elif params["model_name"] == 'GCN_Att':
    model = GCN_Att(**params['model'])
elif params["model_name"] == 'GCN_Att_Drop_Multihead':
    model = GCN_Att_Drop_Multihead(**params['model'])
elif params["model_name"] == 'GCN_Att_Not_res':
    model = GCN_Att_Not_res(**params['model'])
else:
    raise ValueError(f'{params["model_name"]} is not a valid model name')

model = model.to(device)

graph = graph.to(device)


optimizer_gcn = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
criterion = loss_fn_SimCLR
if params["train_contrastive"]:
    model = train_node_embedder(model, graph, optimizer_gcn, criterion, config=params,
                                name_model=f'{run_path}/Weights/contr_sup_{name_yaml}.pth')
else:
    model.load_state_dict(torch.load(f'{run_path}/Weights/contr_sup_{name_yaml}.pth'))

# Get the embeddings of the nodes
model.eval()
with torch.no_grad():
    out = model.contrastive(graph)
    out = out.cpu().numpy()
    labels = graph.y.cpu().numpy()

# Save the embeddings
import pickle as pkl
with open(f'{run_path}/Pickles/embeds_contr_sup_{name_yaml}.pkl', 'wb') as file:
    pkl.dump(out, file)

with open(f"{run_path}/Pickles/train_test_val_masks_{name_yaml}.pkl", "wb") as file:
    pkl.dump([train_mask, val_mask, test_mask, train_mask_contrastive], file)


# Plot the embeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if out.shape[1] > 2:
    # Applying t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(out[3305:])  # Assuming the first 3304 are unlabeled and hence excluded


    # Separating the reduced features by their labels for plotting
    X_tsne_benign = X_tsne[labels[3305:] == 0]
    X_tsne_fraudulent = X_tsne[labels[3305:] == 1]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_drop_{name_yaml}_all.png')
    plt.close()

    # Plotting only the test nodes
    X_tsne_test = X_tsne[graph.test_mask.cpu().numpy()[3305:], :]
    labels_test = labels[3305:][graph.test_mask.cpu().numpy()[3305:]]
    X_tsne_benign = X_tsne_test[labels_test == 0]
    X_tsne_fraudulent = X_tsne_test[labels_test == 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model (Test nodes only)')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_{name_yaml}_test.png')
    plt.close()

    # Plotting only the train nodes
    X_tsne_train = X_tsne[graph.train_mask.cpu().numpy()[3305:], :]
    labels_train = labels[3305:][graph.train_mask.cpu().numpy()[3305:]]
    X_tsne_benign = X_tsne_train[labels_train == 0]
    X_tsne_fraudulent = X_tsne_train[labels_train == 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model (Train nodes only)')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_{name_yaml}_train.png')
    plt.close()
else:
    # As the embeddings are already 2D, we can directly plot them

    X_tsne = out[3305:] # Assuming the first 3304 are unlabeled and hence excluded

    # Separating the reduced features by their labels for plotting
    X_tsne_benign = X_tsne[labels[3305:] == 0]
    X_tsne_fraudulent = X_tsne[labels[3305:] == 1]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_{name_yaml}_all.png')
    plt.close()

    X_tsne = out

    # Plotting only the test nodes
    X_tsne_test = X_tsne[graph.test_mask.cpu().numpy(), :]
    labels_test = labels[graph.test_mask.cpu().numpy()]
    X_tsne_benign = X_tsne_test[labels_test == 0]
    X_tsne_fraudulent = X_tsne_test[labels_test == 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model (Test nodes only)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_{name_yaml}_test.png')
    plt.close()

    # Plotting only the train nodes
    X_tsne_train = X_tsne[graph.train_mask.cpu().numpy(), :]
    labels_train = labels[graph.train_mask.cpu().numpy()]
    X_tsne_benign = X_tsne_train[labels_train == 0]
    X_tsne_fraudulent = X_tsne_train[labels_train == 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model (Train nodes only)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_{name_yaml}_train.png')
    plt.close()


if params["train_head"]:
    # Frozing all the model parameters that are not on the classification head
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer_gcn = torch.optim.AdamW(parameters, lr=params["lr"], weight_decay=params["weight_decay"])

    train_samples = graph.y[graph.train_mask]
    weight_for_class_0 = len(train_samples) / (len(train_samples[train_samples == 0]) * 2)
    weight_for_class_1 = len(train_samples) / (len(train_samples[train_samples == 1]) * 2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([weight_for_class_0, weight_for_class_1]).to(device))
    model = train_node_classifier_minibatches(model=model, graph=graph, config=params, 
                                            criterion=criterion, optimizer=optimizer_gcn, self_supervised=True, 
                                            name_model=f'{run_path}/Weights/head_contr_sup_{name_yaml}.pth')
    model.load_state_dict(torch.load(f'{run_path}/Weights/head_contr_sup_{name_yaml}.pth'))

    test_acc, f1, predictions = eval_node_classifier(model, graph, graph.test_mask)
    print(f'Test Acc: {test_acc:.3f}, Test F1: {f1:.3f}')

    conf_matrix = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(),
                                predictions[graph.test_mask].cpu().numpy())
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{run_path}/Plots/cm_contr_sup_{name_yaml}.png')
    plt.close()

    from sklearn.metrics import classification_report
    report = classification_report(graph.y[graph.test_mask].cpu().numpy(), predictions[graph.test_mask].cpu().numpy(), output_dict=True)
    report["ROC_AUC"] = compute_ROC_AUC(model, graph, graph.test_mask)
    
    with open(f'{run_path}/Report/contr_sup_{name_yaml}.txt', 'w') as file:
        file.write(str(report))