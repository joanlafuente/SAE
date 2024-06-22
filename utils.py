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
import random
import os

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from pytorch_metric_learning import losses, miners, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import wandb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess_data(params, name_yaml, approach, use_percentage_train=1):
    if params["data"] == "amz":
        # Creating the path to save the run files
        run_path = f"./Runs/{approach}/Amazon/{name_yaml}"

        # Creating a folder for the run files
        if not os.path.exists(f'{run_path}'):
            os.makedirs(f'{run_path}')
            os.makedirs(f'{run_path}/Weights')
            os.makedirs(f'{run_path}/Plots')
            os.makedirs(f'{run_path}/Report')
            if approach != 'Supervised':
                os.makedirs(f'{run_path}/Pickles')

        # Loading data
        data_file = loadmat('./Data/Amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A

        # Splitting the data into train, validation and test
        nodes = list(range(3305, 11944))
        train_nodes, test_val_nodes = train_test_split(nodes, train_size=0.6, stratify=labels[nodes], random_state=0)
        val_nodes, test_nodes = train_test_split(test_val_nodes, train_size=0.5, stratify=labels[test_val_nodes], random_state=0)
        # In case not all the training data is used
        if use_percentage_train != 1:
            train_nodes, not_used_nodes = train_test_split(train_nodes, train_size=use_percentage_train, stratify=labels[train_nodes], random_state=0)
            print(f'Using {use_percentage_train * 0.6 * 100}% of the training data')

        # Nodes used for the contrastive learning (Training set + non-labelled data)
        train_nodes_contrastive = train_nodes + list(range(0, 3305))

        # Inizializing the masks for the train, validation and test sets
        train_mask = torch.zeros(11944, dtype=torch.bool)
        val_mask = torch.zeros(11944, dtype=torch.bool)
        test_mask = torch.zeros(11944, dtype=torch.bool)
        train_mask_contrastive = torch.zeros(11944, dtype=torch.bool)

        # Setting the masks
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True
        train_mask_contrastive[train_nodes_contrastive] = True

        # Loading the adjacency lists for the three types of edges
        with open('./Data/amz_upu_adjlists.pickle', 'rb') as file:
            upu = pickle.load(file)
        with open('./Data/amz_usu_adjlists.pickle', 'rb') as file:
            usu = pickle.load(file)
        with open('./Data/amz_uvu_adjlists.pickle', 'rb') as file:
            uvu = pickle.load(file)


        # Processing the adjacency lists to create the edge lists
        # For each type of edge, an edge list is created
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

        # Creating graph data structure from torch_geometric
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
        run_path = f"./Runs/{approach}/Yelp/{name_yaml}"
        # Creating a folder for the run files
        if not os.path.exists(f'{run_path}'):
            os.makedirs(f'{run_path}')
            os.makedirs(f'{run_path}/Weights')
            os.makedirs(f'{run_path}/Plots')
            os.makedirs(f'{run_path}/Report')
            if approach != 'Supervised':
                os.makedirs(f'{run_path}/Pickles')

        # Loading data
        data_file = loadmat('./Data/YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A

        num_nodes = feat_data.shape[0]

        # Splitting the data into train, validation and test
        nodes = np.arange(num_nodes)
        train_nodes, test_val_nodes = train_test_split(nodes, train_size=0.7, stratify=labels, random_state=0)
        val_nodes, test_nodes = train_test_split(test_val_nodes, train_size=0.5, stratify=labels[test_val_nodes], random_state=0)
        # In case not all the training data is used
        if use_percentage_train != 1:
            train_nodes, not_used_nodes = train_test_split(train_nodes, train_size=use_percentage_train, stratify=labels[train_nodes], random_state=0)
            print(f'Using {use_percentage_train * 0.7 * 100}% of the training data')
        # Nodes used for the contrastive learning (In this case is the same as the training set)
        train_nodes_contrastive = train_nodes 

        # Inizializing the masks for the train, validation and test sets
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask_contrastive = torch.zeros(num_nodes, dtype=torch.bool)

        # Setting the masks
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True
        train_mask_contrastive[train_nodes_contrastive] = True

        # Loading the adjacency lists for the three types of edges
        with open('./Data/yelp_rtr_adjlists.pickle', 'rb') as file:
            upu = pickle.load(file)
        with open('./Data/yelp_rsr_adjlists.pickle', 'rb') as file:
            usu = pickle.load(file)
        with open('./Data/yelp_rur_adjlists.pickle', 'rb') as file:
            uvu = pickle.load(file)

        # Processing the adjacency lists to create the edge lists
        # For each type of edge, an edge list is created
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

        # Creating graph data structure from torch_geometric
        graph = Data(x=torch.tensor(feat_data).float(), 
                    edge_index_v=torch.tensor(edges_list_v), 
                    edge_index_p=torch.tensor(edges_list_p),
                    edge_index_s=torch.tensor(edges_list_s),
                    y=torch.tensor(labels).type(torch.int64),
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    train_mask_contrastive=train_mask_contrastive)
    
    return graph, run_path, train_mask, val_mask, test_mask, train_mask_contrastive

def compute_ROC_curve(model, graph, mask):
    model.eval()
    with torch.no_grad():
        # Get the predictions of the model
        pred = model(graph)[mask]
        labels = graph.y[mask]
        # Apply softmax to get the probabilities
        pred = F.softmax(pred, dim=1)
        # Get the probabilities of the positive class (Anomaly class)
        pred = pred[:, 1]
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        # Compute the ROC curve
        fpr, tpr, _ = roc_curve(labels, pred)
    return fpr, tpr

def compute_PR_curve(model, graph, mask):
    model.eval()
    with torch.no_grad():
        # Get the predictions of the model
        pred = model(graph)[mask]
        labels = graph.y[mask]
        # Apply softmax to get the probabilities
        pred = F.softmax(pred, dim=1)
        # Get the probabilities of the positive class (Anomaly class)
        pred = pred[:, 1]
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        # Compute the PR curve
        precision, recall, _ = precision_recall_curve(labels, pred)
    return precision, recall

def compute_ROC_AUC(model, graph, mask):
    model.eval()
    with torch.no_grad():
        # Get the predictions of the model
        pred = model(graph)[mask]
        labels = graph.y[mask]
        # Apply softmax to get the probabilities
        pred = F.softmax(pred, dim=1)
        # Get the probabilities of the positive class (Anomaly class)
        pred = pred[:, 1]
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        return roc_auc_score(labels, pred)#, multi_class='ovr', average='micro')
    
def compute_Average_Precision(model, graph, mask):
    model.eval()
    with torch.no_grad():
        # Get the predictions of the model
        pred = model(graph)[mask]
        labels = graph.y[mask]
        # Apply softmax to get the probabilities
        pred = F.softmax(pred, dim=1)
        # Get the probabilities of the positive class (Anomaly class)
        pred = pred[:, 1]
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        # Compute the average precision
        return average_precision_score(labels, pred)

def eval_node_classifier(model, graph, mask):
    model.eval()
    # Get the predictions of the model
    pred = model(graph).argmax(dim=1)

    # Compute the accuracy
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    # Macro f1 score
    predictions = pred[mask].cpu().numpy()
    y_true = graph.y[mask].cpu().numpy()
    f1 = f1_score(y_true, predictions, average='macro')
    return acc, f1, pred

def mask_node_atributes(graph, percentage_n=0.1, percentage_e=0.1):
    # Masking the node attributes
    drop = nn.Dropout(percentage_n)
    graph = copy.deepcopy(graph)
    graph.x = drop(graph.x)

    # Using bernoulli distribution remove edges (As GraphCL)
    mask_p = torch.bernoulli(torch.full((graph.edge_index_p.size(1),), percentage_e)).bool()
    mask_s = torch.bernoulli(torch.full((graph.edge_index_s.size(1),), percentage_e)).bool()
    mask_v = torch.bernoulli(torch.full((graph.edge_index_v.size(1),), percentage_e)).bool()

    # Masking the edges
    graph.edge_index_p = graph.edge_index_p[:, mask_p]
    graph.edge_index_s = graph.edge_index_s[:, mask_s]
    graph.edge_index_v = graph.edge_index_v[:, mask_v]

    return graph

def augment_graph(graph, percentage_n=0.9, percentage_e=0.1):
    drop = nn.Dropout(percentage_n)
    graph = copy.deepcopy(graph)

    # Add random noise to the features
    noise = torch.randn_like(graph.x)*0.1
    noise = drop(noise)
    graph.x = graph.x + noise


    # Using bernoulli distribution remove edges (As GraphCL)
    mask_p = torch.bernoulli(torch.full((graph.edge_index_p.size(1),), percentage_e)).bool()
    mask_s = torch.bernoulli(torch.full((graph.edge_index_s.size(1),), percentage_e)).bool()
    mask_v = torch.bernoulli(torch.full((graph.edge_index_v.size(1),), percentage_e)).bool()

    # Masking the edges
    graph.edge_index_p = graph.edge_index_p[:, mask_p]
    graph.edge_index_s = graph.edge_index_s[:, mask_s]
    graph.edge_index_v = graph.edge_index_v[:, mask_v]

    return graph

def eval_node_embedder(model, graph, mask, criterion, percentage_n=0.1, percentage_e=0.1):
    # Compute validation the loss of the model for the GraphCL approach
    model.eval()
    with torch.no_grad():
        # Get the predictions of the model applying to different random transformations as in GraphCL
        pred = model.contrastive(mask_node_atributes(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        pred2 = model.contrastive(mask_node_atributes(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        # Compute the loss
        loss = criterion(pred, pred2)
    return loss.cpu().item()

def eval_node_embedder_samp(model, graph, mask, criterion, percentage_n=0.1, percentage_e=0.1):
    # Compute validation the loss of the model for the GraphCL approach variant
    model.eval()
    with torch.no_grad():
        # Get the predictions of the model applying to different random transformations
        pred = model.contrastive(augment_graph(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        pred2 = model.contrastive(augment_graph(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        # Compute the loss
        loss = criterion(pred, pred2)
    return loss.cpu().item()


def train_node_embedder(model, graph, optimizer, criterion, config, name_model='best_model_contrastive.pth'):
    # Get the transformation to apply to the graph
    if config["self_supervised"]["augmentations"] == 'dropout':
        percentage = config["self_supervised"]["percentage_masking_features"]
        percentage_e = config["self_supervised"]["percentage_masking_edges"]
        augment_func = mask_node_atributes
    elif config["self_supervised"]["augmentations"] == 'sampling':
        percentage = config["self_supervised"]["percentage_features_not_modified"]
        percentage_e = config["self_supervised"]["percentage_masking_edges"]
        augment_func = augment_graph
    else:
        raise ValueError(f'{config["self_supervised"]["augmentations"]} is not a valid augmentation type')

    # Init the variables for the early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    # Init the lr squeduler
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
    # Init wandb
    wandb.init(project='Self-supervised contrastive learning', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)
    
    # Load efficient precision at 1 metric
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k=2047)

    batch_size = config["batch_size"]
    for epoch in tqdm(range(1, config["epochs"] + 1)):
        model.train()
        epoch_loss = 0

        # Get the indices to shuffle the embeddings and the labels
        num_train_nodes = graph.train_mask.sum().item()
        indices = torch.randperm(num_train_nodes)
        
        counter = 0
        # Iterating over the embedings and the labels, on batches
        for batch in range(0, graph.train_mask.sum().item(), batch_size):
            # Obtain the embeddings of the nodes applying the two transformations
            out = model.contrastive(augment_func(graph, percentage_n=percentage, percentage_e=percentage_e).to(device))
            out2 = model.contrastive(augment_func(graph, percentage_n=percentage, percentage_e=percentage_e).to(device))

            out = out[graph.train_mask_contrastive][indices][batch:batch+batch_size]
            out2 = out2[graph.train_mask_contrastive][indices][batch:batch+batch_size]
        
            # Skip the batch if the batch is empty
            if len(out) == 0:
                continue

            # Compute the loss and backpropagate
            loss = criterion(out, out2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            counter += 1

        # Compute the validation loss
        model.eval()
        if config["self_supervised"]["augmentations"] == 'dropout':
            val_loss = eval_node_embedder(model, graph, graph.val_mask, criterion, percentage_n=percentage, percentage_e=percentage_e)
        elif config["self_supervised"]["augmentations"] == 'sampling':
            val_loss = eval_node_embedder_samp(model, graph, graph.val_mask, criterion, percentage_n=percentage, percentage_e=percentage_e)

        early_stopping_counter += 1
        squeduler.step()

        # Obtain the embeddings of the nodes withouth any transformation
        # to compute the precision at 1
        embeds = model.contrastive(graph)

        # Compute the precision at 1 for the validation set
        accuracies = accuracy_calculator.get_accuracy(query=embeds[graph.val_mask], query_labels=graph.y[graph.val_mask])
        precision = accuracies["precision_at_1"] 

        # Compute the precision at 1 for the training set
        accuracies_train = accuracy_calculator.get_accuracy(query=embeds[graph.train_mask], query_labels=graph.y[graph.train_mask])
        train_precision = accuracies_train["precision_at_1"]

        # Log the epoch metrics to wandb
        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Val precision at 1': precision, 
                   'Train precision at 1': train_precision, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        # Save the model if the validation loss is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        # If the early stopping epochs is met, stop the training
        if early_stopping_counter > config["early_stopping"]:
            break

    # Load the best model
    model.load_state_dict(torch.load(name_model))
    # Load the precision at 1 metric
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",))
    # Obtain the embeddings of the nodes
    embeds = model.contrastive(graph)
    # Compute the precision at 1 for the test set
    test_precision = accuracy_calculator.get_accuracy(query=embeds[graph.test_mask], query_labels=graph.y[graph.test_mask])["precision_at_1"]
    # Log the precision at 1 for the test set
    wandb.log({'Test precision at 1': test_precision}, step=epoch)
    wandb.finish()
    return model




def loss_fn_SimCLR(proj_1, proj_2):
    batch_size = proj_1.size(0)
    temperature = 0.5

    proj_1 = F.normalize(proj_1, dim=-1)
    proj_2 = F.normalize(proj_2, dim=-1)
    # [2*B, D]
    out = torch.cat([proj_1, proj_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) /  temperature)
    # torch.eye() => creates a diagonal matrix ones in the diagonal
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool() # All ones/True except the diagonal

    # [2*B, 2*B-1]
    # The similaries of the diagonal are removed (Are the maximum always as they are the similary with themselves)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(proj_1 * proj_2, dim=1) /  temperature) 

    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    return ((-torch.log(pos_sim / (sim_matrix.sum(dim=1)))).sum()/2)/batch_size


def train_node_embedder_supervised(model, graph, optimizer, config, 
                                   name_model='best_model_contrastive_sup.pth'):
    labels = graph.y
    
    # Loadin cosine distance
    distance = distances.CosineSimilarity()

    # Triplet loss
    loss_func = losses.TripletMarginLoss(margin = config["triplets"]["loss_margin"], distance = distance) 
    # Mining function to get the triplets, keeping the hard and semi-hard triplets
    mining_func = miners.TripletMarginMiner(margin = config["triplets"]["mining_margin"], distance=distance, type_of_triplets="all")
    
    # Init the variables for the early stopping
    best_val_precision = float('-inf')
    early_stopping_counter = 0
    # Init the lr squeduler
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
    # Init wandb
    wandb.init(project='Supervised contrastive learning', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)
    
    # Load efficient precision at 1 metric
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k=2047)

    batch_size = config["batch_size"]
    for epoch in tqdm(range(1, config["epochs"] + 1)):
        model.train()
        epoch_loss = 0

        # Get the indices to shuffle the embeddings and the labels
        num_train_nodes = graph.train_mask.sum().item()
        indices = torch.randperm(num_train_nodes)
        labels = labels[graph.train_mask][indices]
        
        counter = 0
        # Iterating over the embedings and the labels, on batches
        for batch in range(0, graph.train_mask.sum().item(), batch_size):
            # Obtain the embeddings of the nodes
            embeds = model.contrastive(graph)

            # Shuffle the embeddings and the labels (Same shuffle)
            embeds = embeds[graph.train_mask][indices]
            # Get the embeddings and the labels of the batch
            emb = embeds[batch:batch+batch_size]
            y = labels[batch:batch+batch_size]
            # Skip the batch if the batch is empty
            if len(emb) == 0:
                continue
            # Get the hard and semi-hard triplets
            indices_tuple = mining_func(emb, y)
            # Compute and backpropagate the loss
            loss = loss_func(emb, y, indices_tuple)
            epoch_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
  

        labels = graph.y
        # Compute the validation loss
        model.eval()
        with torch.no_grad():
            # Get the embeddings of the nodes
            embeds = model.contrastive(graph)

            # Iterate over the validation set in batches
            val_loss = 0
            counter_val = 0
            for batch in range(0, graph.val_mask.sum().item(), batch_size):
                # Get the embeddings and the labels of a batch
                emb = embeds[graph.val_mask][batch:batch+batch_size]
                y = graph.y[graph.val_mask][batch:batch+batch_size]

                # Skip the batch if the batch is empty
                if len(emb) == 0:
                    continue

                # Get the hard and semi-hard triplets
                indices_tuple = mining_func(emb, y)
                # Compute the loss
                loss = loss_func(emb, y, indices_tuple)
                val_loss += loss.item()
                counter_val += 1
        # Compute the average loss on the validation set
        val_loss = val_loss/counter_val

        early_stopping_counter += 1
        squeduler.step()

        # Compute training and validation precision at 1
        accuracies = accuracy_calculator.get_accuracy(query=embeds[graph.val_mask], query_labels=graph.y[graph.val_mask])
        precision = accuracies["precision_at_1"]
        accuracies_train = accuracy_calculator.get_accuracy(query=embeds[graph.train_mask], query_labels=graph.y[graph.train_mask])
        train_precision = accuracies_train["precision_at_1"]

        # Log the epoch metrics to wandb
        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Val precision at 1': precision, 
                   'Train precision at 1': train_precision, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        # Save the model if the validation precision at 1 is the best
        if best_val_precision < precision:
            best_val_precision = precision
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        # If the early stopping epochs is met, stop the training
        if early_stopping_counter > config["early_stopping"]:
            break
    
    # Load the best model
    model.load_state_dict(torch.load(name_model))
    # Load the precision at 1 metric
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",))
    # Obtain the embeddings of the nodes
    embeds = model.contrastive(graph)
    # Compute the precision at 1 for the test set
    test_precision = accuracy_calculator.get_accuracy(query=embeds[graph.test_mask], query_labels=graph.y[graph.test_mask])["precision_at_1"]
    # Log the precision at 1 for the test set
    wandb.log({'Test precision at 1': test_precision}, step=epoch)
    wandb.finish()
    return model



def train_node_classifier_minibatches(model, graph, optimizer, config, 
                                      criterion=None, only_head=False,
                                      self_supervised=False,
                                      name_model='best_model_clasifier.pth'):
    labels = graph.y

    # If no loss is provided, use CrossEntropyLoss
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Init the variables for the early stopping
    best_val_f1 = float('-inf')
    early_stopping_counter = 0
    # Init the lr squeduler
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
    # Add keywords to the run name in wandb, used to identify the run on specific cases
    if only_head:
        config["run_name"] += '_triplet_head'
        config["pretrained_triplet"] = True
    if self_supervised:
        config["run_name"] += '_head'
        config["pretrained_self_supervised"] = True


    # Init wandb
    wandb.init(project='Classical Supervised learning', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)

    batch_size = config["batch_size"]
    for epoch in tqdm(range(1, config["epochs"] + 1)):
        model.train()
        epoch_loss = 0

        # Get the indices to shuffle the embeddings and the labels
        num_train_nodes = graph.train_mask.sum().item()
        indices = torch.randperm(num_train_nodes)
        labels = labels[graph.train_mask][indices]
        
        counter = 0
        # Iterating over the embedings and the labels, on batches
        for batch in range(0, len(graph.train_mask), batch_size):
            # Obtain the model predictions
            preds = model(graph)

            # Shuffle the embeddings and the labels (Same shuffle)
            preds = preds[graph.train_mask][indices]
            # Get the embeddings and the labels of the batch
            pred = preds[batch:batch+batch_size]
            y = labels[batch:batch+batch_size]

            # Skip the batch if the batch is empty
            if len(pred) == 0:
                continue

            # Compute the loss and backpropagate
            loss = criterion(pred, y)
            epoch_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
       
        # Compute the validation loss
        labels = graph.y
        model.eval()
        with torch.no_grad():
            # Get the model predictions and compute the loss
            preds = model(graph)
            metric = criterion(preds[graph.val_mask], labels[graph.val_mask])

        early_stopping_counter += 1
        squeduler.step()

        # Compute the f1 score for the validation and training set
        f1_val = f1_score(labels[graph.val_mask].cpu().numpy(), preds[graph.val_mask].argmax(dim=1).cpu().numpy(), average='macro')
        f1_train = f1_score(labels[graph.train_mask].cpu().numpy(), preds[graph.train_mask].argmax(dim=1).cpu().numpy(), average='macro')

        # Log the epoch metrics to wandb
        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': metric, 
                   'Val f1': f1_val, 
                   'Train f1': f1_train, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        # Save the model if the validation f1 is the best
        if best_val_f1 < f1_val:
            best_val_f1 = f1_val
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        # If the early stopping epochs is met, stop the training
        if early_stopping_counter > config["early_stopping"]:
            break
    
    # Load the best model
    model.load_state_dict(torch.load(name_model))

    # Compute the f1 score for the test set
    with torch.no_grad():
        preds = model(graph)
        f1_test = f1_score(labels[graph.test_mask].cpu().numpy(), preds[graph.test_mask].argmax(dim=1).cpu().numpy(), average='macro')
    # Compute the ROC-AUC for the test set
    test_ROC_AUC = compute_ROC_AUC(model, graph, graph.test_mask)

    # Log the f1 score and ROC-AUC for the test set
    wandb.log({'Test f1': f1_test, "Test ROC-AUC":test_ROC_AUC}, step=epoch)
    wandb.finish()
    return model


def train_node_embedder_and_classifier_supervised(model, graph, optimizer, criterion, 
                                                  config, name_model='best_model.pth'):
    labels = graph.y
    # Load the weight given for each loss
    weight_triplet = config["weight_triplet"]
    weight_classification = config["weight_classification"]

    # Load cosine distance
    distance = distances.CosineSimilarity()

    # Triplet loss
    loss_func = losses.TripletMarginLoss(margin = config["triplets"]["loss_margin"], distance = distance) 
    # Mining function to get the triplets, keeping the hard and semi-hard triplets
    mining_func = miners.TripletMarginMiner(margin = config["triplets"]["mining_margin"], distance=distance, type_of_triplets="all")
    
    # Init the variables for the early stopping
    best_val_f1 = float('-inf')
    early_stopping_counter = 0
    # Init the lr squeduler
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
    # Init wandb
    wandb.init(project='Supervised contrastive and classification', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)
    
    # Load efficient precision at 1 metric
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k=2047)

    batch_size = config["batch_size"]
    for epoch in tqdm(range(1, config["epochs"] + 1)):
        model.train()
        epoch_loss = 0

        # Get the indices to shuffle the embeddings and the labels
        num_train_nodes = graph.train_mask.sum().item()
        indices = torch.randperm(num_train_nodes)
        labels = labels[graph.train_mask][indices]
        
        counter = 0
        # Iterating over the embedings and the labels, on batches
        for batch in range(0, graph.train_mask.sum().item(), batch_size):
            # Obtain the embeddings of the nodes
            embeds = model.contrastive(graph)

            # Shuffle the embeddings and the labels (Same shuffle)
            embeds = embeds[graph.train_mask][indices]

            # Get the embeddings and the labels of the batch
            emb = embeds[batch:batch+batch_size]
            y = labels[batch:batch+batch_size]

            # Skip the batch if the batch is empty
            if len(emb) == 0:
                continue

            # Get the hard and semi-hard triplets and compute the loss
            indices_tuple = mining_func(emb, y)
            loss = loss_func(emb, y, indices_tuple) * weight_triplet

            # Compute the model predictions using the predicted embeddings
            pred = model.classifier(emb)
            # Add the classification loss to the total loss
            loss += criterion(pred, y) * weight_classification

            epoch_loss += loss.item()

            # Backpropagate the loss
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        labels = graph.y

        # Compute the validation loss
        model.eval()
        with torch.no_grad():
            # Get the embeddings of the nodes
            embeds = model.contrastive(graph)

            # Iterate over the validation set in batches
            val_loss = 0
            counter_val = 0
            for batch in range(0, graph.val_mask.sum().item(), batch_size):
                # Get the embeddings and the labels of a batch
                emb = embeds[graph.val_mask][batch:batch+batch_size]
                y = graph.y[graph.val_mask][batch:batch+batch_size]

                # Skip the batch if the batch is empty
                if len(emb) == 0:
                    continue

                # Get the hard and semi-hard triplets
                indices_tuple = mining_func(emb, y)

                # Compute the triplet loss and the classification loss
                loss = loss_func(emb, y, indices_tuple) * weight_triplet
                pred = model.classifier(emb)
                loss += criterion(pred, y) * weight_classification
                val_loss += loss.item()
                counter_val += 1

        # Compute the average loss on the validation set
        val_loss = val_loss/counter_val

        early_stopping_counter += 1
        squeduler.step()


        # Compute training and validation precision at 1
        accuracies = accuracy_calculator.get_accuracy(query=embeds[graph.val_mask], query_labels=graph.y[graph.val_mask])
        precision = accuracies["precision_at_1"]
        accuracies_train = accuracy_calculator.get_accuracy(query=embeds[graph.train_mask], query_labels=graph.y[graph.train_mask])
        train_precision = accuracies_train["precision_at_1"]

        # Compute the f1 score for the validation and training set 
        preds = model.classifier(embeds)
        f1_val = f1_score(labels[graph.val_mask].cpu().numpy(), preds[graph.val_mask].argmax(dim=1).cpu().numpy(), average='macro')
        f1_train = f1_score(labels[graph.train_mask].cpu().numpy(), preds[graph.train_mask].argmax(dim=1).cpu().numpy(), average='macro')

        # Log the epoch metrics to wandb
        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Val precision at 1': precision, 
                   'Train precision at 1': train_precision, 
                   'Val f1': f1_val,
                   'Train f1': f1_train,
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        # Save the model if the validation f1 is the best
        if best_val_f1 < f1_val:
            best_val_f1 = f1_val
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        # If the early stopping epochs is met, stop the training
        if early_stopping_counter > config["early_stopping"]:
            break
    
    # Load the best model
    model.load_state_dict(torch.load(name_model))

    # Load the precision at 1 metric
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",))
    # Obtain the embeddings of the nodes
    embeds = model.contrastive(graph)
    # Compute the precision at 1 for the test set
    test_precision = accuracy_calculator.get_accuracy(query=embeds[graph.test_mask], query_labels=graph.y[graph.test_mask])["precision_at_1"]

    # Compute the f1 score for the test set
    with torch.no_grad():
        preds = model(graph)
        test_f1 = f1_score(labels[graph.test_mask].cpu().numpy(), preds[graph.test_mask].argmax(dim=1).cpu().numpy(), average='macro')

    # Compute the ROC-AUC for the test set
    test_ROC_AUC = compute_ROC_AUC(model, graph, graph.test_mask)

    # Log the test metrics to wandb
    wandb.log({'Test f1': test_f1, "Test ROC-AUC":test_ROC_AUC, 'Test precision at 1': test_precision}, step=epoch)
    wandb.finish()
    return model




def train_node_autoencoder(model, graph, optimizer, config, 
                                      label2use=0,
                                      criterion=None, 
                                      name_model='best_model_clasifier.pth'):
    labels = graph.y

    # If no loss is provided, use MSELoss
    if criterion is None:
        criterion = nn.MSELoss()

    # Init the variables for the early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Init the lr squeduler
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')

    # Init wandb
    wandb.init(project='Autoencoder', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)

    batch_size = config["batch_size"]
    for epoch in tqdm(range(1, config["epochs"] + 1)):
        model.train()
        epoch_loss = 0

        # Get the indices to shuffle the embeddings and the labels
        num_train_nodes = graph.train_mask.sum().item()
        indices = torch.randperm(num_train_nodes)
        feat = graph.x[graph.train_mask][indices]
        y = graph.y[graph.train_mask][indices]
        
        counter = 0
        # Iterating over the embedings and the labels, on batches
        for batch in range(0, len(graph.train_mask), batch_size):
            preds = model(graph)

            # Shuffle the embeddings and the labels (Same shuffle than the one on the labels)
            preds = preds[graph.train_mask][indices]

            # Labels of the batch
            y_batch = y[batch:batch+batch_size]

            # Obtain the embeddings of the nodes on the batch if the label is the one to use (The one to reduce the reconstruction error)
            pred = preds[batch:batch+batch_size][y_batch == label2use]
            ft = feat[batch:batch+batch_size][y_batch == label2use]

            # Skip the batch if the batch is empty
            if len(pred) == 0:
                continue

            # Compute the reconstruction loss
            loss = criterion(pred, ft)

            # Pick the embeddings of the nodes on the batch in which the label is not the one that we want to reduce the reconstruction error
            pred = preds[batch:batch+batch_size][y_batch != label2use]
            ft = feat[batch:batch+batch_size][y_batch != label2use]

            # If is set to increase the reconstruction error of the other classes and the batch is not empty
            if (len(pred) > 0) and (config["alpha"] != 0):
                # Compute the reconstruction error of the other classes
                loss2 = criterion(pred, ft) * config["alpha"]

                # Clipping the loss to avoid exploding gradients and increasing the reconstruction error to infinity
                loss2 = torch.clamp(loss2, max=config["cliping_loss_value"])
                # We add a negative sign to the loss, as we want to maximize this part
                # Augment the reconstruction error of the other classes
                loss -= loss2

            # Compute the loss and backpropagate
            epoch_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compute the validation loss
        feat = graph.x
        labels_val = graph.y[graph.val_mask]
        model.eval()
        with torch.no_grad():
            preds = model(graph)
            val_loss = criterion(preds[graph.val_mask][labels_val == label2use], feat[graph.val_mask][labels_val == label2use])

        early_stopping_counter += 1
        squeduler.step()
        
        # Log the epoch metrics to wandb
        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

    
        # Save the model if the validation loss is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0

        # If the patience is -1, save the model at each epoch
        if config["early_stopping"] == -1:
            torch.save(model.state_dict(), name_model)
        # If the early stopping epochs is met, stop the training
        elif early_stopping_counter > config["early_stopping"]:
            break
    
    # Load the best model 
    model.load_state_dict(torch.load(name_model))

    # Compute the test loss
    labels_test = graph.y[graph.test_mask]
    with torch.no_grad():
        test_loss = criterion(model(graph)[graph.test_mask][labels_test == label2use], graph.x[graph.test_mask][labels_test == label2use])

    # Log the test loss
    wandb.log({'Test loss': test_loss}, step=epoch)
    wandb.finish()
    return model


def train_autoencoder_edges(model, graph, optimizer, config, 
                            name_model='best_model_clasifier.pth'):
    # Init the variables for the early stopping
    best_val_loss = float('inf')
    
    # Init wandb
    wandb.init(project='Autoencoder', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)

    for epoch in tqdm(range(1, config["epochs"] + 1)):
        model.train()
        epoch_loss = 0

        # Get the three node representations for each node
        z1, z2, z3 = model.contrastive(graph)

        # Compute the loss (Use each representation to predict each type of edge)
        loss = model.compute_loss((z1, z2, z3), graph, graph.train_mask_contrastive)
        epoch_loss += loss.item()

        # Backpropagate the loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute the validation loss
        val_loss = 0
        model.eval()
        with torch.no_grad():
            # Get the three node representations for each node
            z1, z2, z3 = model.contrastive(graph)

            # Compute the loss (Use each representation to predict each type of edge)
            loss = model.compute_loss((z1, z2, z3), graph, graph.val_mask)
            val_loss += loss.item()


        # Log the epoch metrics to wandb
        wandb.log({'Train loss': epoch_loss,
                   'Val loss': val_loss}, step=epoch)
        
        # Save the model if the validation loss is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0

        # If the patience is -1, save the model at each epoch
        if config["early_stopping"] == -1:
            torch.save(model.state_dict(), name_model)
        # If the early stopping epochs is met, stop the training
        elif early_stopping_counter > config["early_stopping"]:
            break

    # Load the best model
    model.load_state_dict(torch.load(name_model))
    wandb.finish()
    return model
