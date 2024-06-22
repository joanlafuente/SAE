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
from sklearn.metrics import classification_report
import seaborn as sns
import copy
import os
import sys

from utils import *
from models import GCN, Simpler_GCN, PNA_Edge_feat, Simpler_GCN2, Simpler_GCN_Conv, GCN_Att, GCN_Att_Drop_Multihead, GCN_Att_Not_res, GAT_Edge_feat, GAT_BatchNormalitzation, GAT_SELU_Alphadrop, GIN_ReLU, GIN_tanh, GraphSAGE_model, PNA_model, PNA_model_2
import yaml


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


# Load the graph, the masks and the run path
graph, run_path, train_mask, val_mask, test_mask, train_mask_contrastive = preprocess_data(params, "Supervised", name_yaml,  
                                                                                           use_percentage_train=use_percentage_train)

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
elif params["model_name"] == 'PNA_Edge_feat':
    model = PNA_Edge_feat(**params['model'])
else:
    raise ValueError(f'{params["model_name"]} is not a valid model name')

# Move the model into cuda if available
model = model.to(device)
graph = graph.to(device)

# Obtain the parameters that require gradients
parameters = filter(lambda p: p.requires_grad, model.parameters())

# Define the optimizer
optimizer_gcn = torch.optim.AdamW(parameters, lr=params["lr"], weight_decay=params["weight_decay"])

# Compute the class weights for the loss function
# In order to balance the classes weight on the loss function
train_samples = graph.y[graph.train_mask]
weight_for_class_0 = len(train_samples) / (len(train_samples[train_samples == 0]) * 2)
weight_for_class_1 = len(train_samples) / (len(train_samples[train_samples == 1]) * 2)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([weight_for_class_0, weight_for_class_1]).to(device))

if ("OnlyEval" in params.keys()) and params["OnlyEval"]:
    # Load the model if we are only evaluating 
    model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))
    print('Model loaded for evaluation')
elif ("ContrastiveAndClassification" in params.keys()) and params["ContrastiveAndClassification"]:
    # Train the model with the contrastive and classification loss
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

    # Save the masks used for training, validation and testing
    with open(f"{run_path}/train_test_val_masks_{name_yaml}.pkl", "wb") as file:
        pkl.dump([train_mask, val_mask, test_mask, train_mask_contrastive], file)


    # Reduce the dimensionality of the embeddings for visualization
    if ("DimReduction" not in params.keys()) or (params["DimReduction"] == "tsne"):
        # Applying t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(out[3305:])  # Assuming the first 3304 are unlabeled and hence excluded

    elif params["DimReduction"] == "autoencoder":
        # Applying a single layer MLP autoencoder
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

    # Plotting all the nodes
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
else:
    # Train the model with only the classification loss
    model = train_node_classifier_minibatches(model=model, graph=graph, config=params, 
                                          criterion=criterion, optimizer=optimizer_gcn, 
                                          name_model=f'{run_path}/Weights/cls_sup_{name_yaml}.pth')
    
if ("GenerateTSNE" in params.keys()) and params["GenerateTSNE"]:
    # Get the embeddings of the nodes
    model.eval()
    with torch.no_grad():
        out = model.contrastive(graph)
        out = out.cpu().numpy()
        labels = graph.y.cpu().numpy()

    # Applying t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(out[3305:])

    # Separating the reduced features by their labels for plotting
    X_tsne_benign = X_tsne[labels[3305:] == 0]
    X_tsne_fraudulent = X_tsne[labels[3305:] == 1]

    # Plotting all the nodes
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_TSNE_contr_sup_drop_{name_yaml}_all.png')
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
    plt.savefig(f'{run_path}/Plots/embeds_TSNE_contr_sup_{name_yaml}_test.png')
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
    plt.savefig(f'{run_path}/Plots/embeds_TSNE_contr_sup_{name_yaml}_train.png')
    plt.close()

# Load the weights of the previously trained model
model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))

# Evaluate the model
test_acc, f1, predictions = eval_node_classifier(model, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}, Test F1: {f1:.3f}')

# Compute the confusion matrix
conf_matrix = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(),
                               predictions[graph.test_mask].cpu().numpy())
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f'{run_path}/Plots/cm_cls_sup_{name_yaml}.png')
plt.close()

# Compute several metrics for both classes
report = classification_report(graph.y[graph.test_mask].cpu().numpy(), predictions[graph.test_mask].cpu().numpy(), output_dict=True)

# Computing the area under the curve and the average precision scores
report["ROC_AUC"] = compute_ROC_AUC(model, graph, graph.test_mask)
report["AP"] = compute_Average_Precision(model, graph, graph.test_mask)

# Computing the ROC curve
fpr, tpr = compute_ROC_curve(model, graph, graph.test_mask)
# Plotting the ROC curve
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve",
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(f'{run_path}/Plots/roc_curve_{name_yaml}.png')
plt.close()

# Computing the Precision-Recall curve
precision, recall = compute_PR_curve(model, graph, graph.test_mask)

# Plotting the Precision-Recall curve
plt.figure()
lw = 2
plt.plot(
    recall,
    precision,
    color="darkorange",
    lw=lw,
    label="Precision-Recall curve",
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.savefig(f'{run_path}/Plots/PR_curve_{name_yaml}.png')
plt.close()

# Save the classification metrics
with open(f'{run_path}/Report/cls_{name_yaml}.txt', 'w') as file:
    file.write(str(report))