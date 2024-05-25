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
import pickle as pkl


from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import copy
import os
import sys

from utils import *
from models import GCN, Simpler_GCN, Simpler_GCN2, Simpler_GCN_Conv, GCN_Att, GCN_Att_Drop_Multihead, GCN_Att_Not_res, GAT_Edge_feat, GAT_BatchNormalitzation, GAT_SELU_Alphadrop, GIN_ReLU, GIN_tanh, GraphSAGE_model, PNA_model, PNA_model_2
import yaml


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Open a yaml file with the parameters
with open(f'./Setups/Supervised/{name_yaml}.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

if "train_data_percentage" not in params.keys():
    use_percentage_train = 1
else:
    total_train = 0.7 if params["data"] == "yelp" else 0.6
    use_percentage_train = params["train_data_percentage"] / total_train
    if use_percentage_train > 1:
        raise ValueError(f'train_data_percentage cannot be greater than {total_train} for the {params["data"]} dataset')



if params["data"] == "amz":
    run_path = f"./Runs/Supervised/Amazon/{name_yaml}"
    # Creating a folder for the run files
    if not os.path.exists(f'{run_path}'):
        os.makedirs(f'{run_path}')
        os.makedirs(f'{run_path}/Weights')
        os.makedirs(f'{run_path}/Plots')
        os.makedirs(f'{run_path}/Report')

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
    
    if use_percentage_train != 1:
        train_nodes, not_used_nodes = train_test_split(train_nodes, train_size=use_percentage_train, stratify=labels[train_nodes], random_state=0)
        print(f'Using {use_percentage_train * 0.6 * 100}% of the training data')
    
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
    run_path = f"./Runs/Supervised/Yelp/{name_yaml}"
    # Creating a folder for the run files
    if not os.path.exists(f'{run_path}'):
        os.makedirs(f'{run_path}')
        os.makedirs(f'{run_path}/Weights')
        os.makedirs(f'{run_path}/Plots')
        os.makedirs(f'{run_path}/Report')
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
    if use_percentage_train != 1:
        train_nodes, not_used_nodes = train_test_split(train_nodes, train_size=use_percentage_train, stratify=labels[train_nodes], random_state=0)
        print(f'Using {use_percentage_train * 0.7 * 100}% of the training data')
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
elif params["model_name"] == 'GAT_Edge_feat':
    model = GAT_Edge_feat(**params['model'])
elif params["model_name"] == 'GAT_BatchNormalitzation':
    model = GAT_BatchNormalitzation(**params['model'])
elif params["model_name"] == 'GAT_SELU_Alphadrop':
    model = GAT_SELU_Alphadrop(**params['model'])
elif params["model_name"] == 'GIN_ReLU':
    model = GIN_ReLU(**params['model'])
elif params["model_name"] == 'GIN_tanh':
    model = GIN_tanh(**params['model'])
elif params["model_name"] == 'GraphSAGE_model':
    model = GraphSAGE_model(**params['model'])
elif params["model_name"] == 'PNA_model':
    model = PNA_model(**params['model'])
elif params["model_name"] == 'PNA_model_2':
    model = PNA_model_2(**params['model'])
else:
    raise ValueError(f'{params["model_name"]} is not a valid model name')

model = model.to(device)

graph = graph.to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer_gcn = torch.optim.AdamW(parameters, lr=params["lr"], weight_decay=params["weight_decay"])

train_samples = graph.y[graph.train_mask]
weight_for_class_0 = len(train_samples) / (len(train_samples[train_samples == 0]) * 2)
weight_for_class_1 = len(train_samples) / (len(train_samples[train_samples == 1]) * 2)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([weight_for_class_0, weight_for_class_1]).to(device))

if ("ContrastiveAndClassification" in params.keys()) and params["ContrastiveAndClassification"]:
    model = train_node_embedder_and_classifier_supervised(model=model, graph=graph, optimizer=optimizer_gcn, criterion=criterion,
                                                          config=params, name_model=f'{run_path}/Weights/cls_sup_{name_yaml}.pth')
    
    # Get the embeddings of the nodes
    model.eval()
    with torch.no_grad():
        out = model.contrastive(graph)
        out = out.cpu().numpy()
        labels = graph.y.cpu().numpy()

    # Save the embeddings
    with open(f'{run_path}/embeds_contr_sup_{name_yaml}.pkl', 'wb') as file:
        pkl.dump(out, file)

    with open(f"{run_path}/train_test_val_masks_{name_yaml}.pkl", "wb") as file:
        pkl.dump([train_mask, val_mask, test_mask, train_mask_contrastive], file)


    if ("DimReduction" not in params.keys()) or (params["DimReduction"] == "tsne"):
        # Applying t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(out[3305:])  # Assuming the first 3304 are unlabeled and hence excluded

    elif params["DimReduction"] == "autoencoder":
        from sklearn.neural_network import MLPRegressor
        autoencoder = MLPRegressor(hidden_layer_sizes=(2, ), 
                                   activation='identity',
                                   random_state=42)
        
        autoencoder.fit(out[3305:], out[3305:])
        weights = autoencoder.coefs_[0]
        X_tsne = np.dot(out[3305:], weights)

    # Separating the reduced features by their labels for plotting
    X_tsne_benign = X_tsne[labels[3305:] == 0]
    X_tsne_fraudulent = X_tsne[labels[3305:] == 1]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model')
    if ("DimReduction" not in params.keys()) or (params["DimReduction"] == "tsne"):
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
    elif params["DimReduction"] == "autoencoder":
        plt.xlabel('Autoencoder feature 1')
        plt.ylabel('Autoencoder feature 2')
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
    if ("DimReduction" not in params.keys()) or (params["DimReduction"] == "tsne"):
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
    elif params["DimReduction"] == "autoencoder":
        plt.xlabel('Autoencoder feature 1')
        plt.ylabel('Autoencoder feature 2')
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
    if ("DimReduction" not in params.keys()) or (params["DimReduction"] == "tsne"):
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
    elif params["DimReduction"] == "autoencoder":
        plt.xlabel('Autoencoder feature 1')
        plt.ylabel('Autoencoder feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_{name_yaml}_train.png')
    plt.close()


elif ("OnlyEval" in params.keys()) and params["OnlyEval"]:
    model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))
    print('Model loaded for evaluation')
else:
    model = train_node_classifier_minibatches(model=model, graph=graph, config=params, 
                                          criterion=criterion, optimizer=optimizer_gcn, 
                                          name_model=f'{run_path}/Weights/cls_sup_{name_yaml}.pth')

model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))

test_acc, f1, predictions = eval_node_classifier(model, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}, Test F1: {f1:.3f}')

conf_matrix = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(),
                               predictions[graph.test_mask].cpu().numpy())
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f'{run_path}/Plots/cm_cls_sup_{name_yaml}.png')
plt.close()

from sklearn.metrics import classification_report
report = classification_report(graph.y[graph.test_mask].cpu().numpy(), predictions[graph.test_mask].cpu().numpy(), output_dict=True)


report["ROC_AUC"] = compute_ROC_AUC(model, graph, graph.test_mask)
report["AP"] = compute_Average_Precision(model, graph, graph.test_mask)


fpr, tpr = compute_ROC_curve(model, graph, graph.test_mask)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(f'{run_path}/Plots/roc_curve_{name_yaml}.png')
plt.close()


with open(f'{run_path}/Report/cls_{name_yaml}.txt', 'w') as file:
    file.write(str(report))