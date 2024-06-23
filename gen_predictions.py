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
from models import GCN, GAE_model, GAE_model_PNA, Simpler_GCN, Simpler_GCN2, Simpler_GCN_Conv, GCN_Att, GCN_Att_Drop_Multihead, GCN_Att_Not_res, GAT_Edge_feat, GAT_BatchNormalitzation, GAT_SELU_Alphadrop, GIN_ReLU, GIN_tanh, GraphSAGE_model, PNA_model, PNA_model_2
import yaml

"""
Script to generate predictions of a previously trained model.

"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Check if the script has the correct input arguments
if len(sys.argv) != 3:
    raise ValueError('The script needs two arguments: the name of the yaml file and the type of run.\nExample: python gen_predictions.py name_yaml Supervised\n')
if sys.argv[2] not in ["Autoencoder", "SelfSupervisedContrastive", "Supervised", "SupervisedContrastive"]:
    raise ValueError(f'{sys.argv[2]} is not a valid run type. Use Autoencoder, SelfSupervisedContrastive, Supervised or SupervisedContrastive.')

# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Get the name of the type of run
run_type = sys.argv[2]
print(f'Run type: {run_type}')

# Open a yaml file with the parameters
with open(f'./Setups/{run_type}/{name_yaml}.yaml') as file:
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
graph, run_path, train_mask, val_mask, test_mask, train_mask_contrastive = preprocess_data(params, name_yaml, run_type, use_percentage_train=use_percentage_train)


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
elif params["model_name"] == 'GAE_model_PNA':
    model = GAE_model_PNA(**params['model'])
elif params["model_name"] == 'GAE_model':
    model = GAE_model(**params['model'])
else:
    raise ValueError(f'{params["model_name"]} is not a valid model name')

# Load the trained model weights
if os.path.exists(f'{run_path}/Weights/cls_sup_{name_yaml}.pth'):
    model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))
elif os.path.exists(f'{run_path}/Weights/head_contr_sup_{name_yaml}.pth'):
    model.load_state_dict(torch.load(f'{run_path}/Weights/head_contr_sup_{name_yaml}.pth', map_location=device))
else:
    raise ValueError(f'The model {name_yaml} does not exist or has not a classification head trained.\nTrain the model first or if is the case use the GMM classifier.')

# Move the model and graph to cuda if available
model = model.to(device)
graph = graph.to(device)

# Get the predictions of the model
model.eval()
with torch.no_grad():
    pred = model(graph)

# Getting the probability of being an anomaly
softmax_fn = nn.Softmax(dim=1)
pred = softmax_fn(pred)
pred = pred[:, 1]

# Create the folder to save the predictions
if not os.path.exists(f'{run_path}/Predictions'):
    os.makedirs(f'{run_path}/Predictions')

# Saving the predictions
pred = pred.cpu().numpy()
np.save(f'{run_path}/Predictions/predictions.npy', pred)