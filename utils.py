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
import seaborn as sns
import copy


from pytorch_metric_learning import losses, miners, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import wandb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=1000, name_model='best_model.pth'):
    loss_history = []
    f1_history = []
    best_val_f1 = float('-inf')
    counter = 0

    squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.75)

    wandb.init(project='Graph contrastive Learning (Node classification)')
    wandb.watch(model)
    

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred = out.argmax(dim=1)
        model.eval()

        train_acc, train_f1, _ = eval_node_classifier(model, graph, graph.train_mask)
        acc, f1, pred = eval_node_classifier(model, graph, graph.val_mask)
        loss_history.append(loss.item())
        f1_history.append(f1)

        wandb.log({'Train loss': loss.item(),
                    'Train accuracy': train_acc,
                    'Train f1': train_f1,
                    'Val accuracy': acc,
                    'Val f1': f1,
                    'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)


        counter += 1
        squeduler.step()

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), name_model)
            counter = 0

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

        if counter > 300:
            break

    model.load_state_dict(torch.load(name_model))
    test_acc, test_f1, _ = eval_node_classifier(model, graph, graph.test_mask)
    wandb.log({'Test accuracy': test_acc,
               'Test f1': test_f1}, step=epoch)
    wandb.finish()
    return model, loss_history, f1_history


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

def mask_node_atributes(graph, percentage=0.1, percentage_e=0.1):
    # Masking the node attributes
    drop = nn.Dropout(percentage)
    graph = copy.deepcopy(graph)
    graph.x = drop(graph.x)

    # Using bernoulli distribution remove edges (As GraphCL)
    mask_p = torch.bernoulli(torch.full((graph.edge_index_p.size(1),), percentage_e)).bool()
    mask_s = torch.bernoulli(torch.full((graph.edge_index_s.size(1),), percentage_e)).bool()
    mask_v = torch.bernoulli(torch.full((graph.edge_index_v.size(1),), percentage_e)).bool()

    # print(mask_p.sum(), mask_s.sum(), mask_v.sum())
    # print(mask_p.size(), mask_s.size(), mask_v.size())

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

# Compute precision at k, of the contrastive model
def precision_at_k_embeds(embeds, labels, k=1):
    sim = torch.mm(embeds, embeds.t())
    sim = sim - torch.eye(sim.size(0)).to(device)*10
    precision = []
    for i in range(sim.size(0)):
        _, indices = torch.topk(sim[i], k=k)
        retrieved = labels[indices]
        precision.append((retrieved == labels[i]).sum().item()/k)

    precision = np.mean(precision)
    return precision

def train_node_embedder(model, graph, optimizer, criterion, n_epochs=100, percentage=0.35, name_model='best_model_contrastive.pth'):
    loss_history = []
    metric_history = []
    precision_history = []
    best_val_metric = float('inf')
    counter = 0
    squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()

        out = model.contrastive(mask_node_atributes(graph, percentage=percentage).to(device))
        out2 = model.contrastive(mask_node_atributes(graph, percentage=percentage).to(device))

        loss = criterion(out[graph.train_mask_contrastive], out2[graph.train_mask_contrastive])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())
        # print(loss.item())        

        model.eval()
        metric = eval_node_embedder(model, graph, graph.val_mask, criterion, percentage=percentage)
        # print(metric)
        metric_history.append(metric)

        counter += 1
        squeduler.step()

        if metric < best_val_metric:
            best_val_metric = metric
            torch.save(model.state_dict(), name_model)
            counter = 0

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}')
            print(f'Epoch: {epoch:03d}, Val Loss: {metric:.3f}')
            precision = precision_at_k(model, graph, graph.val_mask, k=1)
            precision_history.append(precision)
            print(f'Epoch: {epoch:03d}, Val Precision at 1: {precision:.3f}')

        if counter > 300:
            break

    model.load_state_dict(torch.load(name_model))
    return model, loss_history, metric_history, precision_history



def eval_node_embedder(model, graph, mask, criterion, percentage=0.35):
    model.eval()
    with torch.no_grad():
        pred = model.contrastive(mask_node_atributes(graph, percentage=percentage).to(device))[mask]
        pred2 = model.contrastive(mask_node_atributes(graph, percentage=percentage).to(device))[mask]
        loss = criterion(pred, pred2)
    return loss.cpu().item()

def eval_node_embedder_samp(model, graph, mask, criterion, percentage=0.35):
    model.eval()
    with torch.no_grad():
        pred = model.contrastive(augment_graph(graph, percentage=percentage).to(device))[mask]
        pred2 = model.contrastive(augment_graph(graph, percentage=percentage).to(device))[mask]
        loss = criterion(pred, pred2)
    return loss.cpu().item()


def loss_fn(proj_1, proj_2):
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


def augment_graph(graph, percentage=0.1, percentage_e=0.1):
    drop = nn.Dropout(percentage)
    graph = copy.deepcopy(graph)

    # Add random noise to the features
    noise = torch.randn_like(graph.x)*0.1
    noise = drop(noise)
    graph.x = graph.x + noise


    # Using bernoulli distribution remove edges (As GraphCL)
    mask_p = torch.bernoulli(torch.full((graph.edge_index_p.size(1),), percentage_e)).bool()
    mask_s = torch.bernoulli(torch.full((graph.edge_index_s.size(1),), percentage_e)).bool()
    mask_v = torch.bernoulli(torch.full((graph.edge_index_v.size(1),), percentage_e)).bool()

    # print(mask_p.sum(), mask_s.sum(), mask_v.sum())
    # print(mask_p.size(), mask_s.size(), mask_v.size())

    graph.edge_index_p = graph.edge_index_p[:, mask_p]
    graph.edge_index_s = graph.edge_index_s[:, mask_s]
    graph.edge_index_v = graph.edge_index_v[:, mask_v]

    return graph

def train_node_embedder_samp(model, graph, optimizer, criterion, n_epochs=100, percentage=0.35, name_model='best_model_contrastive.pth'):
    loss_history = []
    metric_history = []
    precision_history = []
    best_val_metric = float('inf')
    counter = 0
    squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()

        out = model.contrastive(augment_graph(graph, percentage=percentage).to(device))
        out2 = model.contrastive(augment_graph(graph, percentage=percentage).to(device))

        loss = criterion(out[graph.train_mask_contrastive], out2[graph.train_mask_contrastive])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())
        # print(loss.item())        

        model.eval()
        metric = eval_node_embedder_samp(model, graph, graph.val_mask, criterion, percentage=percentage)
        # print(metric)
        metric_history.append(metric)

        counter += 1
        squeduler.step()

        if metric < best_val_metric:
            best_val_metric = metric
            torch.save(model.state_dict(), name_model)
            counter = 0

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}')
            print(f'Epoch: {epoch:03d}, Val Loss: {metric:.3f}')
            precision = precision_at_k(model, graph, graph.val_mask, k=1)
            precision_history.append(precision)
            print(f'Epoch: {epoch:03d}, Val Precision at 1: {precision:.3f}')

        if counter > 300:
            break

    model.load_state_dict(torch.load(name_model))
    return model, loss_history, metric_history, precision_history





def train_node_embedder_supervised(model, graph, optimizer, variation=None, criterion=None, mining_func=None, 
                                   n_epochs=100, early_stopping=50, name_model='best_model_contrastive_sup.pth', 
                                   batch_size=320, mining_margin=1, loss_margin=1):
    labels = graph.y
    if criterion is None:
        distance = distances.CosineSimilarity()
        loss_func = losses.TripletMarginLoss(margin=loss_margin, distance = distance) # The margin can be diferent to the one used on the miner -> You define what is a semi-hard and hard triplet

    if mining_func is None:
        distance = distances.CosineSimilarity()
        mining_func = miners.TripletMarginMiner(margin = mining_margin, distance=distance, type_of_triplets="all")
    
    loss_history = []
    metric_history = []
    precision_history = []
    best_val_precision = float('-inf')
    early_stopping_counter = 0
    squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65)

    
    # Check if a class has an atribute
    if hasattr(model, 'out_channels'):
        out_channels = model.out_channels
    else:
        out_channels = 0
    
    if hasattr(model, 'hidden_channels'):
        hidden_channels = model.hidden_channels
    else:
        hidden_channels = 0
    
    if hasattr(model, 'num_layers'):
        num_layers = model.num_layers
    else:
        num_layers = 0
    
    if hasattr(model, 'dropout'):
        dropout = model.dropout
    else:
        dropout = 0

    # Init wandb
    wandb.init(project='Graph contrastive Learning', name=name_model.split('.')[0])
    wandb.watch(model)
    wandb.config.update({'batch_size': batch_size, 
                         'n_epochs': n_epochs, 
                         'loss_margin': loss_margin, 
                         'mining_margin': mining_margin,
                         'early_stopping': early_stopping,
                         'variation': variation,
                         'out_channels': out_channels,
                         'hidden_channels': hidden_channels,
                         'num_layers': num_layers,
                         'dropout': dropout,})

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        epoch_loss = 0

        # Get the indices to shuffle the embeddings and the labels
        num_train_nodes = graph.train_mask.sum().item()
        indices = torch.randperm(num_train_nodes)
        labels = labels[graph.train_mask][indices]
        
        counter = 0
        # Iterating over the embedings and the labels, on batches
        for batch in range(0, len(graph.train_mask), batch_size):
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

        loss_history.append(epoch_loss/counter)
        # print(loss.item())        

        labels = graph.y

        model.eval()
        with torch.no_grad():
            embeds = model.contrastive(graph)
            metric = loss_func(embeds[graph.val_mask], labels[graph.val_mask], mining_func(embeds[graph.val_mask], labels[graph.val_mask]))
        # print(metric)
        metric_history.append(metric)

        early_stopping_counter += 1
        squeduler.step()


        precision = precision_at_k_embeds(embeds[graph.val_mask], labels[graph.val_mask], k=1)
        precision_history.append(precision)
        train_precision = precision_at_k_embeds(embeds[graph.train_mask], labels[graph.train_mask], k=1)

        wandb.log({'Train loss': epoch_loss/counter, 
                   'Val loss': metric, 
                   'Val precision at 1': precision, 
                   'Train precision at 1': train_precision, 
                   'Current lr': optimizer.param_groups[0]['lr']}, step=epoch)

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Loss: {metric:.3f}, Val Precision at 1: {precision:.3f}')

        if best_val_precision < precision:
            best_val_precision = precision
            torch.save(model.state_dict(), name_model)
            early_stopping_counter = 0
        
        if early_stopping_counter > early_stopping:
            break
    

    model.load_state_dict(torch.load(name_model))

    test_precision = precision_at_k(model, graph, graph.test_mask, k=1)
    wandb.log({'Test precision at 1': test_precision}, step=epoch)
    wandb.finish()
    return model



def train_node_classifier_minibatches(model, graph, optimizer, variation=None, criterion=None, n_epochs=100, 
                                      early_stopping=50, batch_size=320,
                                      name_model='best_model_clasifier.pth'):
    labels = graph.y
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = float('-inf')
    early_stopping_counter = 0
    squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65)
    # squeduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.65)

    
    # Check if a class has an atribute
    if hasattr(model, 'out_channels'):
        out_channels = model.out_channels
    else:
        out_channels = 0
    
    if hasattr(model, 'hidden_channels'):
        hidden_channels = model.hidden_channels
    else:
        hidden_channels = 0
    
    if hasattr(model, 'num_layers'):
        num_layers = model.num_layers
    else:
        num_layers = 0
    
    if hasattr(model, 'dropout'):
        dropout = model.dropout
    else:
        dropout = 0

    # Init wandb
    wandb.init(project='Graph contrastive Learning', name=name_model.split('.')[0])
    wandb.watch(model)
    wandb.config.update({'batch_size': batch_size, 
                         'n_epochs': n_epochs, 
                         'early_stopping': early_stopping,
                         'variation': variation,
                         'out_channels': out_channels,
                         'hidden_channels': hidden_channels,
                         'num_layers': num_layers,
                         'dropout': dropout,})

    for epoch in tqdm(range(1, n_epochs + 1)):
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
        
        if early_stopping_counter > early_stopping:
            break
    

    model.load_state_dict(torch.load(name_model))

    with torch.no_grad():
        preds = model(graph)
        f1_test = f1_score(labels[graph.test_mask].cpu().numpy(), preds[graph.test_mask].argmax(dim=1).cpu().numpy(), average='macro')
    wandb.log({'Test f1': f1_test}, step=epoch)
    wandb.finish()
    return model