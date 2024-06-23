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

"""
Script to visualize the attention weights of a trained model.
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Open a yaml file with the parameters
with open(f'./Setups/Supervised/{name_yaml}.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# In case the train_data_percentage is not specified, we use the whole training set
# Otherwise, we use the percentage specified of the training set
if "train_data_percentage" not in params.keys():
    use_percentage_train = 1
else:
    total_train = 0.7 if params["data"] == "yelp" else 0.6
    use_percentage_train = params["train_data_percentage"] / total_train
    if use_percentage_train > 1:
        raise ValueError(f'train_data_percentage cannot be greater than {total_train} for the {params["data"]} dataset')

# Load the graph and the masks
graph, run_path, train_mask, val_mask, test_mask, train_mask_contrastive = preprocess_data(params, "Supervised", name_yaml, use_percentage_train=use_percentage_train)

# Load the specified model
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

# Move the model and graph to cuda if available
model = model.to(device)
graph = graph.to(device)

# Load the weights of the trained model
model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))
model.eval()


# Add a hook to the model to get the attention weights
def attention_hook(module, input, output):
    global attention_weights
    attention_weights = output
model.attention.scoringDot.register_forward_hook(attention_hook)

# Get the output of the model
output = model(graph)

# Removing the extra dimension
attention_weights = attention_weights.squeeze() # [45954, 3]

# Get the attention weights and the labels of the test set
attention_weights = attention_weights[graph.test_mask].detach().cpu().numpy() 
labels_test = graph.y[graph.test_mask].detach().cpu().numpy()


# Making an histogram of the attention weights for each type of edge
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


# Making an histogram of the attention weights for each type of edge but now separated by class
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