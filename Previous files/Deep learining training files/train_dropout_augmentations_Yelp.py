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
from models import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)




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


model = GCN_Att_Drop_Multihead(dropout=0.1, in_channels=feat_data.shape[1], hidden_channels=30, out_channels=15)
model = model.to(device)

graph = graph.to(device)

optimizer_gcn = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=5e-4)
criterion = loss_fn
model, loss_hist, metric_hist, precision_hist = train_node_embedder(model, graph, optimizer_gcn, criterion, percentage=0.1, n_epochs=1000, name_model='best_model_contrastive_dropout_augmentations_Yelp.pth')

test_loss = eval_node_embedder(model, graph, graph.test_mask, criterion)
print(f'Test Acc: {test_loss:.3f}')

plt.plot(loss_hist, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('loss_contrastive_dropout_augmentations_Yelp.png')
plt.close()

plt.plot(metric_hist, label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('val_loss_contrastive_dropout_augmentations_Yelp.png')
plt.close()

plt.plot(precision_hist, label='Precision at 1')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('precision_contrastive_dropout_augmentations_Yelp.png')
plt.close()

print('Precision at 1:', precision_at_k(model, graph, graph.test_mask, k=1))


# Get the embeddings of the nodes
model.eval()
with torch.no_grad():
    out = model.contrastive(graph)
    out = out.cpu().numpy()
    labels = graph.y.cpu().numpy()

# Plot the embeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(out[3305:])  # Assuming the first 3304 are unlabeled and hence excluded

import pickle as pkl
with open('embeddings_contrastive_dropout_augmentations_Yelp.pkl', 'wb') as file:
    pkl.dump(out, file)

with open("train_test_val_masks_dropout_augmentations_Yelp.pkl", "wb") as file:
    pkl.dump([train_mask, val_mask, test_mask, train_mask_contrastive], file)

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
plt.savefig('embeddings_contrastive_dropout_augmentations_Yelp.png')
plt.close()

# # Frozing all the model parameters that are not on the classification head
# for name, param in model.named_parameters():
#     if 'classifier' not in name:
#         param.requires_grad = False

# parameters = filter(lambda p: p.requires_grad, model.parameters())

# optimizer_gcn = torch.optim.AdamW(parameters, lr=0.01, weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.095, 0.905]).to(device))
# model, loss_hist_cls, f1_hist_cls = train_node_classifier(model, graph, optimizer_gcn, criterion, n_epochs=1000, name_model='best_model_head_dropout_augmentations_Yelp.pth')

# test_acc, f1, predictions = eval_node_classifier(model, graph, graph.test_mask)
# print(f'Test Acc: {test_acc:.3f}, Test F1: {f1:.3f}')

# plt.plot(loss_hist_cls, label='Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend()
# plt.savefig('loss_classifier_dropout_augmentations_Yelp.png')
# plt.close()

# plt.plot(f1_hist_cls, label='F1 Score')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend()
# plt.savefig('f1_classifier_dropout_augmentations_Yelp.png')
# plt.close()

# conf_matrix = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(),
#                                predictions[graph.test_mask].cpu().numpy())
# sns.heatmap(conf_matrix, annot=True, fmt='d')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.savefig('confusion_matrix_dropout_augmentations_Yelp.png')
# plt.close()

# from sklearn.metrics import classification_report
# report = classification_report(graph.y[graph.test_mask].cpu().numpy(), predictions[graph.test_mask].cpu().numpy())
# print(report)