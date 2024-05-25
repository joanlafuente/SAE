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

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from pytorch_metric_learning import losses, miners, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import wandb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_ROC_curve(model, graph, mask):
    model.eval()
    with torch.no_grad():
        pred = model(graph)[mask]
        labels = graph.y[mask]
        pred = F.softmax(pred, dim=1)
        # Get the probabilities of the positive class
        pred = pred[:, 1]
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        fpr, tpr, _ = roc_curve(labels, pred)
    
    return fpr, tpr

def compute_ROC_AUC(model, graph, mask):
    model.eval()
    with torch.no_grad():
        pred = model(graph)[mask]
        labels = graph.y[mask]
        pred = F.softmax(pred, dim=1)
        # Get the probabilities of the positive class
        pred = pred[:, 1]
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        return roc_auc_score(labels, pred)#, multi_class='ovr', average='micro')
    
def compute_Average_Precision(model, graph, mask):
    model.eval()
    with torch.no_grad():
        pred = model(graph)[mask]
        labels = graph.y[mask]
        pred = F.softmax(pred, dim=1)
        # Get the probabilities of the positive class
        pred = pred[:, 1]
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        return average_precision_score(labels, pred)

def eval_node_classifier(model, graph, mask):

    model.eval()
    pred = model(graph).argmax(dim=1)
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

    graph.edge_index_p = graph.edge_index_p[:, mask_p]
    graph.edge_index_s = graph.edge_index_s[:, mask_s]
    graph.edge_index_v = graph.edge_index_v[:, mask_v]

    return graph

# Compute precision at k, of the contrastive model
def precision_at_k(model, graph, mask, k=1):
    model.eval()
    with torch.no_grad():
        out = model.contrastive(graph)
        out = out[mask]
        labels = graph.y[mask]
        sim = torch.mm(out, out.t())
        sim = sim - torch.eye(sim.size(0)).to(device)*10
        precision = []
        for i in range(sim.size(0)):
            _, indices = torch.topk(sim[i], k=k)
            retrieved = labels[indices]
            precision.append((retrieved == labels[i]).sum().item()/k)

        precision = np.mean(precision)
    return precision

def eval_node_embedder(model, graph, mask, criterion, percentage_n=0.1, percentage_e=0.1):
    model.eval()
    with torch.no_grad():
        pred = model.contrastive(mask_node_atributes(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        pred2 = model.contrastive(mask_node_atributes(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        loss = criterion(pred, pred2)
    return loss.cpu().item()

def eval_node_embedder_samp(model, graph, mask, criterion, percentage_n=0.1, percentage_e=0.1):
    model.eval()
    with torch.no_grad():
        pred = model.contrastive(augment_graph(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        pred2 = model.contrastive(augment_graph(graph, percentage_n=percentage_n, percentage_e=percentage_e).to(device))[mask]
        loss = criterion(pred, pred2)
    return loss.cpu().item()


def train_node_embedder(model, graph, optimizer, criterion, config, name_model='best_model_contrastive.pth'):
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

    best_val_loss = float('inf')
    early_stopping_counter = 0
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
    # Init wandb
    wandb.init(project='Self-supervised contrastive learning', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)
    
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
            out = model.contrastive(augment_func(graph, percentage_n=percentage, percentage_e=percentage_e).to(device))
            out2 = model.contrastive(augment_func(graph, percentage_n=percentage, percentage_e=percentage_e).to(device))

            out = out[graph.train_mask_contrastive][indices][batch:batch+batch_size]
            out2 = out2[graph.train_mask_contrastive][indices][batch:batch+batch_size]
        
            if len(out) == 0:
                continue

            loss = criterion(out, out2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            counter += 1

        # print(loss.item())        

        model.eval()
        if config["self_supervised"]["augmentations"] == 'dropout':
            val_loss = eval_node_embedder(model, graph, graph.val_mask, criterion, percentage_n=percentage, percentage_e=percentage_e)
        elif config["self_supervised"]["augmentations"] == 'sampling':
            val_loss = eval_node_embedder_samp(model, graph, graph.val_mask, criterion, percentage_n=percentage, percentage_e=percentage_e)

        early_stopping_counter += 1
        squeduler.step()


        embeds = model.contrastive(graph)
        # This function takes embeddings and labels and returns a dictionary of the calculated accuracies
        accuracies = accuracy_calculator.get_accuracy(query=embeds[graph.val_mask], query_labels=graph.y[graph.val_mask])
        precision = accuracies["precision_at_1"]

        accuracies_train = accuracy_calculator.get_accuracy(query=embeds[graph.train_mask], query_labels=graph.y[graph.train_mask])
        train_precision = accuracies_train["precision_at_1"]



        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Val precision at 1': precision, 
                   'Train precision at 1': train_precision, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        if early_stopping_counter > config["early_stopping"]:
            break

    model.load_state_dict(torch.load(name_model))
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",))
    embeds = model.contrastive(graph)
    test_precision = accuracy_calculator.get_accuracy(query=embeds[graph.test_mask], query_labels=graph.y[graph.test_mask])["precision_at_1"]
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
    
    distance = distances.CosineSimilarity()

    # The margin can be diferent to the one used on the miner -> You define what is a semi-hard and hard triplet
    loss_func = losses.TripletMarginLoss(margin = config["triplets"]["loss_margin"], distance = distance) 
    mining_func = miners.TripletMarginMiner(margin = config["triplets"]["mining_margin"], distance=distance, type_of_triplets="all")
    
    best_val_precision = float('-inf')
    early_stopping_counter = 0
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
    # Init wandb
    wandb.init(project='Supervised contrastive learning', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)
    
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
            embeds = model.contrastive(graph)

            # Shuffle the embeddings and the labels (Same shuffle)
            embeds = embeds[graph.train_mask][indices]

            emb = embeds[batch:batch+batch_size]
            y = labels[batch:batch+batch_size]
            if len(emb) == 0:
                continue
            indices_tuple = mining_func(emb, y)
            loss = loss_func(emb, y, indices_tuple)
            epoch_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # print(loss.item())        

        labels = graph.y

        model.eval()
        with torch.no_grad():
            embeds = model.contrastive(graph)
            val_loss = 0
            counter_val = 0
            for batch in range(0, graph.val_mask.sum().item(), batch_size):
                emb = embeds[graph.val_mask][batch:batch+batch_size]
                y = graph.y[graph.val_mask][batch:batch+batch_size]
                if len(emb) == 0:
                    continue
                indices_tuple = mining_func(emb, y)
                loss = loss_func(emb, y, indices_tuple)
                val_loss += loss.item()
                counter_val += 1
        val_loss = val_loss/counter_val
        # print(metric)

        early_stopping_counter += 1
        squeduler.step()


        # This function takes embeddings and labels and returns a dictionary of the calculated accuracies
        accuracies = accuracy_calculator.get_accuracy(query=embeds[graph.val_mask], query_labels=graph.y[graph.val_mask])
        precision = accuracies["precision_at_1"]

        accuracies_train = accuracy_calculator.get_accuracy(query=embeds[graph.train_mask], query_labels=graph.y[graph.train_mask])
        train_precision = accuracies_train["precision_at_1"]



        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Val precision at 1': precision, 
                   'Train precision at 1': train_precision, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        if best_val_precision < precision:
            best_val_precision = precision
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        if early_stopping_counter > config["early_stopping"]:
            break
    

    model.load_state_dict(torch.load(name_model))
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",))
    embeds = model.contrastive(graph)
    test_precision = accuracy_calculator.get_accuracy(query=embeds[graph.test_mask], query_labels=graph.y[graph.test_mask])["precision_at_1"]
    wandb.log({'Test precision at 1': test_precision}, step=epoch)
    wandb.finish()
    return model



def train_node_classifier_minibatches(model, graph, optimizer, config, 
                                      criterion=None, only_head=False,
                                      self_supervised=False,
                                      name_model='best_model_clasifier.pth'):
    labels = graph.y
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = float('-inf')
    early_stopping_counter = 0
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
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
            preds = model(graph)

            # Shuffle the embeddings and the labels (Same shuffle)
            preds = preds[graph.train_mask][indices]

            pred = preds[batch:batch+batch_size]
            y = labels[batch:batch+batch_size]
            if len(pred) == 0:
                continue

            loss = criterion(pred, y)

            epoch_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # print(loss.item())        

        labels = graph.y
        model.eval()
        with torch.no_grad():
            preds = model(graph)
            metric = criterion(preds[graph.val_mask], labels[graph.val_mask])

        early_stopping_counter += 1
        squeduler.step()


        f1_val = f1_score(labels[graph.val_mask].cpu().numpy(), preds[graph.val_mask].argmax(dim=1).cpu().numpy(), average='macro')
        f1_train = f1_score(labels[graph.train_mask].cpu().numpy(), preds[graph.train_mask].argmax(dim=1).cpu().numpy(), average='macro')

        
        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': metric, 
                   'Val f1': f1_val, 
                   'Train f1': f1_train, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

    
        if best_val_f1 < f1_val:
            best_val_f1 = f1_val
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        if early_stopping_counter > config["early_stopping"]:
            break
    

    model.load_state_dict(torch.load(name_model))

    with torch.no_grad():
        preds = model(graph)
        f1_test = f1_score(labels[graph.test_mask].cpu().numpy(), preds[graph.test_mask].argmax(dim=1).cpu().numpy(), average='macro')

    test_ROC_AUC = compute_ROC_AUC(model, graph, graph.test_mask)
    wandb.log({'Test f1': f1_test, "Test ROC-AUC":test_ROC_AUC}, step=epoch)
    wandb.finish()
    return model


def train_node_embedder_and_classifier_supervised(model, graph, optimizer, criterion, 
                                                  config, name_model='best_model.pth'):
    labels = graph.y
    weight_triplet = config["weight_triplet"]
    weight_classification = config["weight_classification"]

    distance = distances.CosineSimilarity()

    # The margin can be diferent to the one used on the miner -> You define what is a semi-hard and hard triplet
    loss_func = losses.TripletMarginLoss(margin = config["triplets"]["loss_margin"], distance = distance) 
    mining_func = miners.TripletMarginMiner(margin = config["triplets"]["mining_margin"], distance=distance, type_of_triplets="all")
    
    best_val_f1 = float('-inf')
    early_stopping_counter = 0
    if config["squeduler"]["type"] == 'StepLR':
        squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["squeduler"]["step_size"], gamma=config["squeduler"]["gamma"])
    else:
        raise ValueError(f'{config["squeduler"]["type"]} is not a valid squeduler type')
    
    # Init wandb
    wandb.init(project='Supervised contrastive and classification', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)
    
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
            embeds = model.contrastive(graph)

            # Shuffle the embeddings and the labels (Same shuffle)
            embeds = embeds[graph.train_mask][indices]

            emb = embeds[batch:batch+batch_size]
            y = labels[batch:batch+batch_size]
            if len(emb) == 0:
                continue
            indices_tuple = mining_func(emb, y)
            loss = loss_func(emb, y, indices_tuple) * weight_triplet

            pred = model.classifier(emb)
            loss += criterion(pred, y) * weight_classification

            epoch_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # print(loss.item())        

        labels = graph.y

        model.eval()
        with torch.no_grad():
            embeds = model.contrastive(graph)
            val_loss = 0
            counter_val = 0
            for batch in range(0, graph.val_mask.sum().item(), batch_size):
                emb = embeds[graph.val_mask][batch:batch+batch_size]
                y = graph.y[graph.val_mask][batch:batch+batch_size]
                if len(emb) == 0:
                    continue

                indices_tuple = mining_func(emb, y)
                loss = loss_func(emb, y, indices_tuple) * weight_triplet
                pred = model.classifier(emb)
                loss += criterion(pred, y) * weight_classification
                val_loss += loss.item()
                counter_val += 1
        val_loss = val_loss/counter_val
        # print(metric)

        early_stopping_counter += 1
        squeduler.step()


        # This function takes embeddings and labels and returns a dictionary of the calculated accuracies
        accuracies = accuracy_calculator.get_accuracy(query=embeds[graph.val_mask], query_labels=graph.y[graph.val_mask])
        precision = accuracies["precision_at_1"]

        accuracies_train = accuracy_calculator.get_accuracy(query=embeds[graph.train_mask], query_labels=graph.y[graph.train_mask])
        train_precision = accuracies_train["precision_at_1"]

        preds = model.classifier(embeds)
        f1_val = f1_score(labels[graph.val_mask].cpu().numpy(), preds[graph.val_mask].argmax(dim=1).cpu().numpy(), average='macro')
        f1_train = f1_score(labels[graph.train_mask].cpu().numpy(), preds[graph.train_mask].argmax(dim=1).cpu().numpy(), average='macro')



        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Val precision at 1': precision, 
                   'Train precision at 1': train_precision, 
                   'Val f1': f1_val,
                   'Train f1': f1_train,
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        if best_val_f1 < f1_val:
            best_val_f1 = f1_val
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        if early_stopping_counter > config["early_stopping"]:
            break
    

    model.load_state_dict(torch.load(name_model))
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",))
    embeds = model.contrastive(graph)
    test_precision = accuracy_calculator.get_accuracy(query=embeds[graph.test_mask], query_labels=graph.y[graph.test_mask])["precision_at_1"]

    with torch.no_grad():
        preds = model(graph)
        test_f1 = f1_score(labels[graph.test_mask].cpu().numpy(), preds[graph.test_mask].argmax(dim=1).cpu().numpy(), average='macro')

    test_ROC_AUC = compute_ROC_AUC(model, graph, graph.test_mask)
    wandb.log({'Test f1': test_f1, "Test ROC-AUC":test_ROC_AUC, 'Test precision at 1': test_precision}, step=epoch)
    wandb.finish()
    return model




def train_node_autoencoder(model, graph, optimizer, config, 
                                      label2use=0,
                                      criterion=None, 
                                      name_model='best_model_clasifier.pth'):
    labels = graph.y
    if criterion is None:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')
    early_stopping_counter = 0
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

            # Shuffle the embeddings and the labels (Same shuffle)
            preds = preds[graph.train_mask][indices]

            y_batch = y[batch:batch+batch_size]

            pred = preds[batch:batch+batch_size][y_batch == label2use]
            ft = feat[batch:batch+batch_size][y_batch == label2use]
            if len(pred) == 0:
                continue

            loss = criterion(pred, ft)

            pred = preds[batch:batch+batch_size][y_batch != label2use]
            ft = feat[batch:batch+batch_size][y_batch != label2use]
            if (len(pred) > 0) and (config["alpha"] != 0):
                
                loss2 = criterion(pred, ft) * config["alpha"]

                # Clipping the loss to avoid exploding gradients
                loss2 = torch.clamp(loss2, max=config["cliping_loss_value"])
                # We add a negative sign to the loss, as we want to maximize this part
                # Augment the reconstruction error of the other classes
                loss -= loss2

            epoch_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # print(loss.item())        

        feat = graph.x
        labels_val = graph.y[graph.val_mask]
        model.eval()
        with torch.no_grad():
            preds = model(graph)
            val_loss = criterion(preds[graph.val_mask][labels_val == label2use], feat[graph.val_mask][labels_val == label2use])

        early_stopping_counter += 1
        squeduler.step()
        
        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': val_loss, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

    
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0

        if config["early_stopping"] == -1:
            torch.save(model.state_dict(), name_model)

        elif early_stopping_counter > config["early_stopping"]:
            break
    

    model.load_state_dict(torch.load(name_model))

    labels_test = graph.y[graph.test_mask]
    with torch.no_grad():
        test_loss = criterion(model(graph)[graph.test_mask][labels_test == label2use], graph.x[graph.test_mask][labels_test == label2use])
    wandb.log({'Test loss': test_loss}, step=epoch)
    wandb.finish()
    return model


def train_autoencoder_edges(model, graph, optimizer, config, 
                            name_model='best_model_clasifier.pth'):
    best_val_loss = float('inf')
    
    # Init wandb
    wandb.init(project='Autoencoder', name=config["run_name"])
    wandb.watch(model)
    wandb.config.update(config)

    batch_size = config["batch_size"]
    for epoch in tqdm(range(1, config["epochs"] + 1)):
        model.train()
        epoch_loss = 0

        z1, z2, z3 = model.contrastive(graph)


        loss = model.compute_loss((z1, z2, z3), graph, graph.train_mask_contrastive)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        val_loss = 0
        counter_val = 0
        model.eval()
        with torch.no_grad():
            z1, z2, z3 = model.contrastive(graph)

            loss = model.compute_loss((z1, z2, z3), graph, graph.val_mask)
            val_loss += loss.item()
                # counter_val += 1




        val_loss = val_loss#/counter_val
        epoch_loss = epoch_loss#/counter
        wandb.log({'Train loss': epoch_loss,
                   'Val loss': val_loss}, step=epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0

        if config["early_stopping"] == -1:
            torch.save(model.state_dict(), name_model)

        elif early_stopping_counter > config["early_stopping"]:
            break

    model.load_state_dict(torch.load(name_model))
    wandb.finish()
    return model
