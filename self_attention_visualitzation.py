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
    params["train_data_percentage"] = 0.6 if params["data"] == "amz" else 0.7

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
    train_nodes, test_val_nodes = train_test_split(nodes, train_size=params["train_data_percentage"], stratify=labels[nodes], random_state=0)
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
    train_nodes, test_val_nodes = train_test_split(nodes, train_size=params["train_data_percentage"], stratify=labels, random_state=0)
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


model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))
model.eval()


# Add a hook to the model to get the attention weights
def attention_hook(module, input, output):
    global attention_weights
    attention_weights = output

model.attention.scoringDot.register_forward_hook(attention_hook)

# Get the output of the model
output = model(graph)

attention_weights = attention_weights.squeeze() # [45954, 3]

# Test attention weights
attention_weights = attention_weights[graph.test_mask].detach().cpu().numpy() 

labels_test = graph.y[graph.test_mask].detach().cpu().numpy()


n_bins = 80
fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].hist(attention_weights[:, 0], bins=n_bins, alpha=0.4)
axs[0].set_title('First type of edge')

axs[1].hist(attention_weights[:, 1], bins=n_bins, alpha=0.4)
axs[1].set_title('Second type of edge')

axs[2].hist(attention_weights[:, 2], bins=n_bins, alpha=0.4)
axs[2].set_title('Third type of edge')

plt.suptitle('Distribution of the attention weights')
plt.savefig(f'{run_path}/Plots/attention_weights_distribution.png')
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].hist(attention_weights[:, 0][labels_test == 0], bins=n_bins, alpha=0.4, label='Non-fraudulent', density=True)
axs[0].hist(attention_weights[:, 0][labels_test == 1], bins=n_bins, alpha=0.4, label='Fraudulent', density=True)
axs[0].legend()
axs[0].set_title('First type of edge')

axs[1].hist(attention_weights[:, 1][labels_test == 0], bins=n_bins, alpha=0.4, label='Non-fraudulent', density=True)
axs[1].hist(attention_weights[:, 1][labels_test == 1], bins=n_bins, alpha=0.4, label='Fraudulent', density=True)
axs[1].legend()
axs[1].set_title('Second type of edge')

axs[2].hist(attention_weights[:, 2][labels_test == 0], bins=n_bins, alpha=0.4, label='Non-fraudulent', density=True)
axs[2].hist(attention_weights[:, 2][labels_test == 1], bins=n_bins, alpha=0.4, label='Fraudulent', density=True)
axs[2].legend()
axs[2].set_title('Third type of edge')

plt.suptitle('Distribution of the attention weights')
plt.savefig(f'{run_path}/Plots/attention_weights_distribution_by_class.png')
plt.close()