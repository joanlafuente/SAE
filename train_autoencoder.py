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

from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import copy
import os
import yaml
import sys
import pickle as pkl
from sklearn.manifold import TSNE

from utils import *
from models import GCN_Att_Not_res_Autoencoder, GAE_model, GAE_model_GAT, GAE_model_PNA

"""
The script is used to train a model on the Yelp or Amazon 
dataset for anomaly detection.

There are to pipelines implented to train the model:
    - Train an edge autoencoder and then the classification head
    - Train an node autoencoder
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Open a yaml file with the parameters
with open(f'./Setups/Autoencoder/{name_yaml}.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# Load the data and get the path of the run (It also creates the folder structure to save the experiment results)
graph, run_path, train_mask, val_mask, test_mask, train_mask_contrastive = preprocess_data(params, name_yaml, "Autoencoder")

# Load the specified model
if params["model_name"] == 'GCN_Att_Not_res_Autoencoder':
    model = GCN_Att_Not_res_Autoencoder(**params['model'])
elif params["model_name"] == 'GAE_model':
    model = GAE_model(**params['model'])
elif params["model_name"] == 'GAE_model_GAT':
    model = GAE_model_GAT(**params['model'])
elif params["model_name"] == 'GAE_model_PNA':
    model = GAE_model_PNA(**params['model'])
else:
    raise ValueError(f'{params["model_name"]} is not a valid model name')


# Move the model into cuda if available
model = model.to(device)
graph = graph.to(device)

# Create the optimizer
optimizer_gcn = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

if "train_edge_autoencoder" in params and params["train_edge_autoencoder"]:
    # Train the edge autoencoder
    model = train_autoencoder_edges(model, graph, optimizer_gcn, 
                                    config=params,
                                    name_model=f'{run_path}/Weights/contr_sup_{name_yaml}.pth')

    # Frozing all the model parameters that are not on the classification head
    for name, param in model.named_parameters():
        if 'encoder' in name:
            print("Frozing", name)
            param.requires_grad = False
    
    # Optain the parametrs that are not frozed 
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Create the optimizer
    optimizer_gcn = torch.optim.AdamW(parameters, lr=params["lr"], weight_decay=params["weight_decay"])

    # Compute the class weights, in order to balance the classes in cross entropy loss
    train_samples = graph.y[graph.train_mask]
    weight_for_class_0 = len(train_samples) / (len(train_samples[train_samples == 0]) * 2)
    weight_for_class_1 = len(train_samples) / (len(train_samples[train_samples == 1]) * 2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([weight_for_class_0, weight_for_class_1]).to(device))

    # Train the classification head
    model = train_node_classifier_minibatches(model=model, graph=graph, config=params, 
                                          criterion=criterion, optimizer=optimizer_gcn, 
                                          name_model=f'{run_path}/Weights/cls_sup_{name_yaml}.pth')
    
    # Get the embeddings of the nodes
    model.eval()
    with torch.no_grad():
        out = model.encode(graph)
        out = out.cpu().numpy()
        labels = graph.y.cpu().numpy()

    # Save the embeddings
    with open(f'{run_path}/Pickles/embeds_contr_sup_{name_yaml}.pkl', 'wb') as file:
        pkl.dump(out, file)

    # Save the masks used for training, validation and testing
    with open(f"{run_path}/Pickles/train_test_val_masks_{name_yaml}.pkl", "wb") as file:
        pkl.dump([train_mask, val_mask, test_mask, train_mask_contrastive], file)

    # Use a single layer MLp as an autoencoder to reduce the dimensionality of the embeddings
    from sklearn.neural_network import MLPRegressor
    autoencoder = MLPRegressor(hidden_layer_sizes=(2, ), 
                                activation='identity',
                                random_state=42)
    
    autoencoder.fit(out[3305:], out[3305:])

    # Applying the autoencoder to obtain the reduced dimensionality features
    weights = autoencoder.coefs_[0]
    X_tsne = np.dot(out[3305:], weights)

    # Separating the reduced features by their labels for plotting
    X_tsne_benign = X_tsne[labels[3305:] == 0]
    X_tsne_fraudulent = X_tsne[labels[3305:] == 1]

    # Plotting all the nodes features
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_benign[:, 0], X_tsne_benign[:, 1], label='Benign (Class 0)', alpha=0.5)
    plt.scatter(X_tsne_fraudulent[:, 0], X_tsne_fraudulent[:, 1], label='Fraudulent (Class 1)', alpha=0.5)
    plt.title('t-SNE visualization of the node embeddings generated by the contrastive model')
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
    plt.xlabel('Autoencoder feature 1')
    plt.ylabel('Autoencoder feature 2')
    plt.legend()
    plt.savefig(f'{run_path}/Plots/embeds_contr_sup_{name_yaml}_train.png')
    plt.close()

    # Load the trained model
    model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))

    # Evaluate the model and obtain the predictions
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

    # Compute the classification report, containing precision, recall, f1-score per class
    report = classification_report(graph.y[graph.test_mask].cpu().numpy(), predictions[graph.test_mask].cpu().numpy(), output_dict=True)
    # Compute the area under the ROC curve
    report["ROC_AUC"] = compute_ROC_AUC(model, graph, graph.test_mask)

    # Compute the ROC curve
    fpr, tpr = compute_ROC_curve(model, graph, graph.test_mask)

    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(f'{run_path}/Plots/roc_curve_{name_yaml}.png')
    plt.close()

    # Save the classification report
    with open(f'{run_path}/Report/cls_{name_yaml}.txt', 'w') as file:
        file.write(str(report))

elif params["train_autoencoder"]:
    # Train the node autoencoder
    model = train_node_autoencoder(model, graph, optimizer_gcn, 
                                        config=params,
                                        label2use=0,
                                        name_model=f'{run_path}/Weights/contr_sup_{name_yaml}.pth')
else:
    # Load a previously trained model
    model.load_state_dict(torch.load(f'{run_path}/Weights/contr_sup_{name_yaml}.pth', map_location=device))
    print("Model loaded")

if ("GenerateTSNE" in params.keys()) and params["GenerateTSNE"]:
    # Get the embeddings of the nodes
    model.eval()
    with torch.no_grad():
        out = model.encode(graph)
        out = out.cpu().numpy()
        labels = graph.y.cpu().numpy()

    # Applying t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(out[3305:])

    # Separating the reduced features by their labels for plotting
    X_tsne_benign = X_tsne[labels[3305:] == 0]
    X_tsne_fraudulent = X_tsne[labels[3305:] == 1]

    # Plotting all the nodes features
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

if ("getEmbeds" in params and params["getEmbeds"]) and not ("train_edge_autoencoder" in params and params["train_edge_autoencoder"]):
    model.eval()
    out = model(graph).detach()

    out = out.cpu().numpy()
    # Save the embeddings
    with open(f'{run_path}/Pickles/embeds_contr_sup_{name_yaml}.pkl', 'wb') as file:
        pkl.dump(out, file)

    with open(f"{run_path}/Pickles/train_test_val_masks_{name_yaml}.pkl", "wb") as file:
        pkl.dump([train_mask, val_mask, test_mask, train_mask_contrastive], file)

elif ("getEmbeds" in params and params["getEmbeds"]) and ("train_edge_autoencoder" in params):
    model.load_state_dict(torch.load(f'{run_path}/Weights/cls_sup_{name_yaml}.pth', map_location=device))
    # Get the embeddings of the nodes
    model.eval()
    with torch.no_grad():
        out = model.encode(graph)
        out = out.cpu().numpy()
        labels = graph.y.cpu().numpy()

    # Save the embeddings
    with open(f'{run_path}/Pickles/embeds_contr_sup_{name_yaml}.pkl', 'wb') as file:
        pkl.dump(out, file)


if params["searchBestTreshold"]:
    # Get the embeddings of the nodes
    out = model(graph).detach()
    train_out = out[graph.train_mask]
    val_out = out[graph.val_mask]
    test_out = out[graph.test_mask]
    DO_ROC = True
    if "BestTreshold" in params and params["BestTreshold"]["PCA"]:
        # Reducing the dimensionality of the embeeding with a PCA
        DO_ROC = False
        from sklearn.decomposition import PCA
        pca = PCA(n_components=params["BestTreshold"]["n_components"])
        train_out = pca.fit_transform(train_out.cpu().numpy())
        val_out = pca.transform(val_out.cpu().numpy())
        test_out = pca.transform(test_out.cpu().numpy())

        print("PCA explained variance ratio:", pca.explained_variance_ratio_)

        # Plot the embeddings after PCA
        if params["BestTreshold"]["n_components"] == 2:
            # If the PCA has 2 components, plot the scatter plot
            plt.scatter(train_out[:, 0], train_out[:, 1], c=graph.y[graph.train_mask].cpu().numpy())
            plt.colorbar()
            plt.savefig(f'{run_path}/Plots/PCA_train.png')
            plt.close()

            plt.scatter(test_out[:, 0], test_out[:, 1], c=graph.y[graph.test_mask].cpu().numpy())
            plt.colorbar()
            plt.savefig(f'{run_path}/Plots/PCA_test.png')
            plt.close()
        elif params["BestTreshold"]["n_components"] == 1:
            # If the PCA has 1 component, plot the histogram
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
        # If the features are not reduced, calculate the errors of the autoencoder, 
        # otherwise use the features as predicted values for the anomaly detection
        if "BestTreshold" in params and params["BestTreshold"]["n_components"] == 1:
            errors_train = train_out
            errors_val = val_out
            errors_test = test_out
        else:
            # Load MSE
            criterion = nn.MSELoss()

            # Compute the MSE of the autoencoder for each node
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
    
        # Compute the z-scores to normalize the data
        mean = errors_train.mean()
        std = errors_train.std()
        errors_train = (errors_train - mean) / std
        errors_val = (errors_val - mean) / std
        errors_test = (errors_test - mean) / std

        # Get the labels
        labels = graph.y[graph.train_mask].cpu().numpy()

        # Use ROC curve to find the best threshold
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
        # Use a KNN classifier to classify the nodes
        knn = KNeighborsClassifier()
        parameters = {'n_neighbors': range(1, 100)}
        # Use grid search to find the best parameters
        clf = GridSearchCV(knn, parameters, cv=5, scoring='f1_macro', verbose=10)

        # Join train and val as we are doing cross validation
        train_val = np.concatenate((train_out, val_out), axis=0)
        labels = np.concatenate((graph.y[graph.train_mask].cpu().numpy(), graph.y[graph.val_mask].cpu().numpy()), axis=0)
        clf.fit(train_val, labels)

        # Get the best parameters
        print(clf.best_params_)
        knn = clf.best_estimator_
        preds = knn.predict(test_out)

        # Get the confidence of each node being an anomaly
        confidence = knn.predict_proba(test_out)[:, 1]
        # Compute the auc score 
        auc = roc_auc_score(graph.y[graph.test_mask].cpu().numpy(), confidence)

    # Plot the confusion matrix
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
    else:
        report["AUC"] = auc

    # Save the metrics report
    with open(f'{run_path}/Report/report.txt', 'w') as file:
        file.write(str(report))