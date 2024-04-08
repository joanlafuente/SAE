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
from models import GCN, Simpler_GCN, Simpler_GCN_Conv, GCN_Att


# Parameters CONTRASTIVE LEARNING
hidden_channels = 35 # If the model has not have attention mechanism, is not used
out_channels = 30
dropout = 0.1
lr = 0.001
epochs_contr = 500
early_stopping = 300
batch_size_contr = 150
variation = 'GCN_Att' # If it is not set on saved files is Simpler_GCN
train_contrastive = True

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
elif variation == 'GCN_Att':
    model = GCN_Att(dropout=dropout, hidden_channels=hidden_channels, out_channels=out_channels)

model = model.to(device)

graph = graph.to(device)

if train_contrastive:
    optimizer_gcn = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    model = train_node_embedder_supervised(model, graph, optimizer_gcn, 
                                            batch_size=batch_size_contr, 
                                            n_epochs=epochs_contr, 
                                            early_stopping=early_stopping,
                                            variation=variation,
                                            name_model=f'./Weights/contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.pth')
else:
    model.load_state_dict(torch.load(f'./Weights/contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.pth'))

# Get the embeddings of the nodes
model.eval()
with torch.no_grad():
    out = model.contrastive(graph)
    out = out.cpu().numpy()
    labels = graph.y.cpu().numpy()

# Save the embeddings
import pickle as pkl
with open(f'./Pickles/embeds_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.pkl', 'wb') as file:
    pkl.dump(out, file)

with open(f"./Pickles/train_test_val_masks__contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.pkl", "wb") as file:
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
    plt.savefig(f'./Plots/embeds_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}_all.png')
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
    plt.savefig(f'./Plots/embeds_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}_test.png')
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
    plt.savefig(f'./Plots/embeds_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}_train.png')
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
    plt.savefig(f'./Plots/embeds_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}_all.png')
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
    plt.savefig(f'./Plots/embeds_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}_test.png')
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
    plt.savefig(f'./Plots/embeds_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}_train.png')
    plt.close()


# Frozing all the model parameters that are not on the classification head
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer_gcn = torch.optim.AdamW(parameters, lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.095, 0.905]).to(device))
model, loss_hist_cls, f1_hist_cls = train_node_classifier(model, graph, optimizer_gcn, criterion, n_epochs=1000, name_model=f'./Weights/head_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.pth')

test_acc, f1, predictions = eval_node_classifier(model, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}, Test F1: {f1:.3f}')

conf_matrix = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(),
                               predictions[graph.test_mask].cpu().numpy())
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f'./Plots/cm_contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.png')
plt.close()

from sklearn.metrics import classification_report
report = classification_report(graph.y[graph.test_mask].cpu().numpy(), predictions[graph.test_mask].cpu().numpy(), output_dict=True)

with open(f'./Reports/contr_sup_drop={dropout}_hidd={hidden_channels}_out={out_channels}_lr={lr}_model={variation}.txt', 'w') as file:
    file.write(str(report))