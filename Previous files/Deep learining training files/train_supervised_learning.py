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

from utils import *
from models import GCN, Simpler_GCN, Simpler_GCN2, Simpler_GCN_Conv, GCN_Att


# Parameters
hidden_channels = 40 # If the model has not have attention mechanism, is not used
out_channels = 40
dropout = 0.5
lr = 0.001
epochs = 500
early_stopping = 50
batch_size = 150
variation = 'GCN_Att' # If it is not set, on saved files, is Simpler_GCN


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)




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


if variation == 'Simpler_GCN':
    model = Simpler_GCN(dropout=dropout, hidden_channels=hidden_channels, out_channels=out_channels)
elif variation == 'Simpler_GCN_Conv':
    model = Simpler_GCN_Conv(dropout=dropout, out_channels=out_channels)
elif variation == 'Simpler_GCN2':
    model = Simpler_GCN2(dropout=dropout, hidden_channels=hidden_channels, out_channels=out_channels)
elif variation == 'GCN_Att':
    model = GCN_Att(dropout=dropout, hidden_channels=hidden_channels, out_channels=out_channels)

model = model.to(device)

graph = graph.to(device)

parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer_gcn = torch.optim.AdamW(parameters, lr=lr, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.095, 0.905]).to(device))

model = train_node_classifier_minibatches(model=model, graph=graph, optimizer=optimizer_gcn, criterion=criterion,
                                                                      n_epochs=epochs, early_stopping=early_stopping, batch_size=batch_size,
                                                                      name_model=f'./Weights/cls_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.pth')

test_acc, f1, predictions = eval_node_classifier(model, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}, Test F1: {f1:.3f}')

conf_matrix = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(),
                               predictions[graph.test_mask].cpu().numpy())
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f'./Plots/cm_cls_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.png')
plt.close()

from sklearn.metrics import classification_report
report = classification_report(graph.y[graph.test_mask].cpu().numpy(), predictions[graph.test_mask].cpu().numpy(), output_dict=True)

with open(f'./Reports/cls_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.txt', 'w') as file:
    file.write(str(report))