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
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import copy
import os
import yaml
import sys

from utils import *
from models import GCN_Att_Not_res_Autoencoder


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Open a yaml file with the parameters
with open(f'./Setups/Autoencoder/{name_yaml}.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

if params["data"] == "amz":
    run_path = f"./Runs/Autoencoder/Amazon/{name_yaml}"
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
    run_path = f"./Runs/Autoencoder/Yelp/{name_yaml}"
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


if params["model_name"] == 'GCN_Att_Not_res_Autoencoder':
    model = GCN_Att_Not_res_Autoencoder(**params['model'])
else:
    raise ValueError(f'{params["model_name"]} is not a valid model name')

model = model.to(device)

graph = graph.to(device)


optimizer_gcn = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
if params["train_autoencoder"]:
    model = train_node_autoencoder(model, graph, optimizer_gcn, 
                                        config=params,
                                        label2use=0,
                                        name_model=f'{run_path}/Weights/contr_sup_{name_yaml}.pth')
else:
    model.load_state_dict(torch.load(f'{run_path}/Weights/contr_sup_{name_yaml}.pth', map_location=device))

if params["searchBestTreshold"]:
    out = model(graph).detach()

    train_out = out[graph.train_mask]
    val_out = out[graph.val_mask]
    test_out = out[graph.test_mask]
    DO_ROC = True
    if "BestTreshold" in params and params["BestTreshold"]["PCA"]:
        DO_ROC = False
        from sklearn.decomposition import PCA
        pca = PCA(n_components=params["BestTreshold"]["n_components"])
        train_out = pca.fit_transform(train_out.cpu().numpy())
        val_out = pca.transform(val_out.cpu().numpy())
        test_out = pca.transform(test_out.cpu().numpy())

        print("PCA explained variance ratio:", pca.explained_variance_ratio_)

        if params["BestTreshold"]["n_components"] == 2:
            plt.scatter(train_out[:, 0], train_out[:, 1], c=graph.y[graph.train_mask].cpu().numpy())
            plt.colorbar()
            plt.savefig(f'{run_path}/Plots/PCA_train.png')
            plt.close()

            plt.scatter(test_out[:, 0], test_out[:, 1], c=graph.y[graph.test_mask].cpu().numpy())
            plt.colorbar()
            plt.savefig(f'{run_path}/Plots/PCA_test.png')
            plt.close()
        elif params["BestTreshold"]["n_components"] == 1:
            DO_ROC = False
            plt.hist(train_out[graph.y[graph.train_mask] == 0], bins=100, alpha=0.5, label='Normal')
            plt.hist(train_out[graph.y[graph.train_mask] == 1], bins=100, alpha=0.5, label='Anomaly')
            plt.legend()
            plt.title('Histogram of values first component of PCA train')
            plt.savefig(f'{run_path}/Plots/Histogram_PCA_train.png')
            plt.close()

            plt.hist(test_out[graph.y[graph.test_mask] == 0], bins=100, alpha=0.5, label='Normal')
            plt.hist(test_out[graph.y[graph.test_mask] == 1], bins=100, alpha=0.5, label='Anomaly')
            plt.legend()
            plt.title('Histogram of values first component of PCA test')
            plt.savefig(f'{run_path}/Plots/Histogram_PCA_test.png')
            plt.close()

            
    if DO_ROC:
        if "BestTreshold" in params and params["BestTreshold"]["n_components"] == 1:
            errors_train = train_out
            errors_val = val_out
            errors_test = test_out
        else:
            # Load MSE
            criterion = nn.MSELoss()

            errors_train, errors_val, errors_test = [], [], []

            for i in range(len(train_out)):
                errors_train.append(criterion(train_out[i], graph.x[i]).item())
                
            for i in range(len(val_out)):
                errors_val.append(criterion(val_out[i], graph.x[i]).item())

            for i in range(len(test_out)):
                errors_test.append(criterion(test_out[i], graph.x[i]).item())

            errors_train = np.array(errors_train)
            errors_val = np.array(errors_val)
            errors_test = np.array(errors_test)
    
        # Compute the z-scores
        mean = errors_train.mean()
        std = errors_train.std()
        errors_train = (errors_train - mean) / std
        errors_val = (errors_val - mean) / std
        errors_test = (errors_test - mean) / std

        labels = graph.y[graph.train_mask].cpu().numpy()

        # Use ROC curve to find the best threshold
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, errors_train)

        # Find the best threshold, which minimize fpr and maximize tpr
        idx = np.argmin(fpr**2 + (1-tpr)**2)
        best_threshold = thresholds[idx]

        # Plot ROC curve
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(f'{run_path}/Plots/ROC.png')
        plt.close()

        # Plot the histogram of errors
        plt.hist(errors_train[labels == 0], bins=100, alpha=0.5, label='Normal')
        plt.hist(errors_train[labels == 1], bins=100, alpha=0.5, label='Anomaly')
        plt.axvline(best_threshold, color='red', label='Threshold')
        plt.legend()
        plt.title('Histogram of Errors')
        plt.savefig(f'{run_path}/Plots/Histogram.png')
        plt.close()

        # Calculate the confusion matrix
        preds = errors_test > best_threshold
    
    else:
        from sklearn.neighbors import KNeighborsClassifier
        # Grid search 
        from sklearn.model_selection import GridSearchCV
        knn = KNeighborsClassifier()
        parameters = {'n_neighbors': range(1, 100)}
        clf = GridSearchCV(knn, parameters, cv=5, scoring='f1_macro', verbose=10)

        train_val = np.concatenate((train_out, val_out), axis=0)
        labels = np.concatenate((graph.y[graph.train_mask].cpu().numpy(), graph.y[graph.val_mask].cpu().numpy()), axis=0)
        clf.fit(train_val, labels)
        print(clf.best_params_)
        knn = clf.best_estimator_
        preds = knn.predict(test_out)


    cm = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(), preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{run_path}/Plots/Confusion_Matrix.png')
    plt.close()

    # Calculate the metrics
    report = classification_report(graph.y[graph.test_mask].cpu().numpy(), preds, output_dict=True)
    print(report)

    if DO_ROC:
        report["AUC"] = roc_auc_score(graph.y[graph.test_mask].cpu().numpy(), errors_test)

    # Save the report
    with open(f'{run_path}/Report/report.txt', 'w') as file:
        file.write(str(report))