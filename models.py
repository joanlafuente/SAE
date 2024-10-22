import torch
from torch import nn
from torch_geometric.nn import GCNConv, GAT, GIN, GAE, GraphSAGE, PNA
from torch_geometric.utils import negative_sampling

"""
Script containing the models used in the experiments.
"""


class SelfAttention2(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Linear projections for the query, key and value
        self.fc_q = nn.Linear(input_size, hidden_size) 
        self.fc_k = nn.Linear(embed_size, hidden_size) 
        self.fc_v = nn.Linear(embed_size, hidden_size) 

        # Initzializing the scoring function for the attention mechanism
        self.scoringDot = scoringDot(self.dropout, self.fc_q, self.fc_k, self.fc_v)

    def forward(self, embeds, node_feat):
        # embeds: batch, 3, embed_size
        # node_feat: batch, input_size

        # Projecting the embeddings to obtain the values
        values = torch.tanh(self.fc_v(self.dropout(embeds))) # Batch, 3, hidden_size

        # Getting the attention weights with the scoring function
        weights = self.scoringDot(embeds, node_feat) # batch, 1, 3

        # Doing the weighted sum of the values
        result = torch.bmm(weights, values) # batch, 1, hidden_size
        result = result.squeeze(1) # batch, hidden_size
        return result
    

class scoringDot(nn.Module):
    def __init__(self, dropout, fc_q, fc_k, fc_v):
        super().__init__()
        # Saving the linear layers and the dropout layer passed as arguments
        self.dropout = dropout
        self.fc_q = fc_q
        self.fc_k = fc_k
        self.fc_v = fc_v

    def forward(self, keys, query):
        # query: batch, input_size
        # keys: batch, 3, embed_size

        # Projecting the initial node features to obtain the query
        query = torch.tanh(self.fc_q(self.dropout(query))) # batch, hidden_size

        # Projecting the embeddings to obtain the keys
        keys = torch.tanh(self.fc_k(self.dropout(keys)))  # batch, 3, hidden_size

        # Obtainin the scores with the dot product
        query = query.unsqueeze(2) # batch, hidden_size, 1
        result = torch.bmm(keys, query) # batch, 3, 1
        result = result.squeeze(2)

        # Normalizing the scores with softmax
        weights = result.softmax(1) # batch, 3
        return weights.unsqueeze(1) # batch, 1, 3  This represents the weights of each type of edge

class PNA_model(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_PNA=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)

        # Parameters of the PNA model 
        # In this case three different mean message passings are done simultaneously for each type of edge
        aggregators = ['mean', 'mean', 'mean']
        scalers = ['identity', 'identity', 'identity']
        deg = torch.tensor([2, 2, 2])

        # Initialize the PNA model for each type of edge
        self.gin_p = PNA(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        self.gin_s = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        
        self.gin_v = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )

        # Initialize the attention mechanism
        self.attention = SelfAttention2(in_channels, out_channels, out_channels)

        # Initialize the classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    def load_state_dict(self, state_dict):
        # Function to load the weights of the model (for perviouly trained models that did not have the scoringDot class)
        if 'attention.scoringDot.fc_q.weight' not in state_dict:
            state_dict['attention.scoringDot.fc_q.weight'] = state_dict['attention.fc_q.weight']
            state_dict['attention.scoringDot.fc_q.bias'] = state_dict['attention.fc_q.bias']

            state_dict['attention.scoringDot.fc_k.weight'] = state_dict['attention.fc_k.weight']
            state_dict['attention.scoringDot.fc_k.bias'] = state_dict['attention.fc_k.bias']

            state_dict['attention.scoringDot.fc_v.weight'] = state_dict['attention.fc_v.weight']
            state_dict['attention.scoringDot.fc_v.bias'] = state_dict['attention.fc_v.bias']
            super().load_state_dict(state_dict)

        else:
            super().load_state_dict(state_dict)

    def contrastive(self, data):
        # Function to obtain the node representations with the trained model

        # Load graph data
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        # Batch normalization of the input features
        x = self.in_norm(x)

        # Appling the PNA model for each type of edge, followed by batch normalization and an activation function
        x_p = self.gin_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gin_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gin_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        # Obtain the node representations
        x = self.contrastive(data)
        # Apply the classification head
        x = self.classifier(x)
        return x
    


class GCN(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.activation = nn.LeakyReLU(0.1)

        # Inizialize the GAT model for each type of edge
        self.gat_p = GAT(in_channels=25, 
                        hidden_channels=150, 
                        num_layers=1,
                        out_channels=50,
                        v2=True)
        self.gat_s = GAT(in_channels=25,
                        hidden_channels=150,
                        num_layers=1,
                        out_channels=50,
                        v2=True)
        self.gat_v = GAT(in_channels=25,
                        hidden_channels=150,
                        num_layers=1,
                        out_channels=50,
                        v2=True)

        # Projection layer to join the embeddings of the three types of edges
        self.projection = nn.Sequential(
            nn.Linear(150, 150),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(150, 75),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(75, 30),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(30, 2)
        )

    def contrastive(self, data):
        # Function to obtain the node representations with the trained model

        # Load graph data
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        # Appling the GAT model for each type of edge, followed by an activation function
        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concatenating the three embeddings
        x = torch.cat((x_p, x_s, x_v), 1)
        # Applying the projection layer
        x = self.projection(x)
        return x

    def forward(self, data):
        # Obtain the node representations
        x = self.contrastive(data)
        # Apply the classification head
        x = self.classifier(x)
        return x
    


class Simpler_GCN(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.activation = nn.LeakyReLU(0.1)

        self.gat_p = GAT(in_channels=25, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_s = GAT(in_channels=25,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_v = GAT(in_channels=25,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels*3, out_channels*2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels*2, out_channels),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        x = torch.cat((x_p, x_s, x_v), 1)
        x = self.projection(x)
        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        x = torch.cat((x_p, x_s, x_v), 1)
        x = self.projection(x)
        x = self.classifier(x)
        return x

class GCN_Att(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25):
        super().__init__()
        self.variation = 'GCN_Att'
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.activation = nn.LeakyReLU(0.1)

        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)

        self.attention = SelfAttention(in_channels, out_channels, out_channels)

        self.projection = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh()
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x_res = self.attention(x_embed, x)

        x = self.projection(x) + x_res

        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x_res = self.attention(x_embed, x)

        x = self.projection(x) + x_res

        x = self.classifier(x)
        return x
    
class GCN_Att_Not_res(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25):
        super().__init__()
        self.variation = 'GCN_Att_Not_res'
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.activation = nn.LeakyReLU(0.1)

        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)

        self.attention = SelfAttention(in_channels, out_channels, out_channels)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        x = self.classifier(x)
        return x
    

class GCN_Att_Drop_Multihead(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, heads=3):
        super().__init__()
        self.variation = 'GCN_Att_Drop_Multihead'
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        print(f'heads: {heads}')
        print(f'hidden_channels: {hidden_channels}')
        print(f'out_channels: {out_channels}')
        print(f'in_channels: {in_channels}')
        print(f'dropout: {dropout}')
        

        self.activation = nn.LeakyReLU(0.1)

        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=heads,
                        concat=True,
                        v2=True)
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=heads,
                        concat=True,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=heads,
                        concat=True,
                        v2=True)

        self.attention = SelfAttention(in_channels, out_channels, out_channels, dropout=dropout)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        x = self.classifier(x)
        return x
    
class GCN_Att_Drop_Multihead_2_Message(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25):
        super().__init__()
        self.variation = 'GCN_Att_Drop_Multihead'
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.activation = nn.LeakyReLU(0.1)

        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=3,
                        concat=True,
                        v2=True)
        self.gat_p2 = GAT(in_channels=out_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=3,
                        concat=True,
                        v2=True)
        
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=3,
                        concat=True,
                        v2=True)
        self.gat_s2 = GAT(in_channels=out_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=3,
                        concat=True,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=3,
                        concat=True,
                        v2=True)
        self.gat_v2 = GAT(in_channels=out_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        dropout=dropout,
                        out_channels=out_channels,
                        heads=3,
                        concat=True,
                        v2=True)

        self.attention = SelfAttention(in_channels, out_channels, out_channels, dropout=dropout)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)
        x_p = x_p + self.gat_p2(x_p, edge_index_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)
        x_s = x_s + self.gat_s2(x_s, edge_index_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)
        x_v = x_v + self.gat_v2(x_v, edge_index_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        x = self.classifier(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.fc_q = nn.Linear(input_size, hidden_size) 
        self.fc_k = nn.Linear(embed_size, hidden_size) 
        self.fc_v = nn.Linear(embed_size, hidden_size) 

    def scoringDot(self, keys, query):
        # query: batch, input_size
        # keys: batch, 3, embed_size
        query = torch.tanh(self.fc_q(self.dropout(query))) # batch, hidden_size
        keys = torch.tanh(self.fc_k(self.dropout(keys)))  # batch, 3, hidden_size

        query = query.unsqueeze(2) # batch, hidden_size, 1
        result = torch.bmm(keys, query) # batch, 3, 1
        result = result.squeeze(2)
        weights = result.softmax(1) # batch, 3
        return weights.unsqueeze(1) # batch, 1, 3  This represents the weights of each type of edge

    def forward(self, embeds, node_feat):
        # embeds: batch, 3, embed_size
        # node_feat: batch, input_size

        values = torch.tanh(self.fc_v(self.dropout(embeds))) # Batch, 3, hidden_size
        weights = self.scoringDot(embeds, node_feat) # batch, 1, 3

        result = torch.bmm(weights, values) # batch, 1, hidden_size
        result = result.squeeze(1) # batch, hidden_size
        return result

class Simpler_GCN2(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=3):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.activation = nn.LeakyReLU(0.1)

        self.gat_p = GAT(in_channels=25, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_s = GAT(in_channels=25,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_v = GAT(in_channels=25,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels*3, out_channels*2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels*2, out_channels),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        x = torch.cat((x_p, x_s, x_v), 1)
        x = self.projection(x)
        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        x = torch.cat((x_p, x_s, x_v), 1)
        x = self.projection(x)
        x = self.classifier(x)
        return x
    
class Simpler_GCN_Conv(torch.nn.Module):
    def __init__(self, dropout=0, out_channels=5):
        super().__init__()
        self.dropout = dropout
        self.out_channels = out_channels

        self.activation = nn.LeakyReLU(0.1)
      
        self.conv_p = GCNConv(25, out_channels)

        self.conv_s = GCNConv(25, out_channels)

        self.conv_v = GCNConv(25, out_channels)

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels*3, out_channels*2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels*2, out_channels),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.conv_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.conv_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.conv_v(x, edge_index_v)
        x_v = self.activation(x_v)

        x = torch.cat((x_p, x_s, x_v), 1)
        x = self.projection(x)
        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.conv_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.conv_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.conv_v(x, edge_index_v)
        x_v = self.activation(x_v)

        x = torch.cat((x_p, x_s, x_v), 1)
        x = self.projection(x)
        x = self.classifier(x)
        return x

class GCN_Att_Not_res_Autoencoder(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels_GNN=5, num_layers=1, in_channels=25):
        super().__init__()
        self.variation = 'GCN_Att_Not_res_Autoencoder'

        self.activation = nn.LeakyReLU(0.1)

        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels_GNN,
                        v2=True)
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels_GNN,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels_GNN,
                        v2=True)

        self.attention = SelfAttention(in_channels, out_channels_GNN, out_channels_GNN)


        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels_GNN, int((out_channels_GNN + in_channels)/2)),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(int((out_channels_GNN + in_channels)/2), in_channels),
        )


    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        x = self.projection(x)
        return x
    

class GAT_Edge_feat(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels_GAT=5,  out_channels_proj=5, num_layers=1, in_channels=25):
        super().__init__()
        self.variation = 'GAT_Edge_feat'

        # TanH activation
        self.activation = nn.Tanh()

        self.gat = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels_GAT,
                        v2=True)
        
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels_GAT, out_channels_proj),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(out_channels_proj, out_channels_proj),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels_proj, int((out_channels_proj+2)/2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int((out_channels_proj+2)/2), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        edge_index = torch.cat((edge_index_p, edge_index_s, edge_index_v), 1)
        edge_attr = torch.cat((torch.ones(edge_index_p.size(1)), torch.ones(edge_index_s.size(1))*2, torch.ones(edge_index_v.size(1))*3)).long()

        x = self.gat(x, edge_index, edge_attr)
        x = self.activation(x)
        x = self.projection(x)
        return x

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v

        edge_index = torch.cat((edge_index_p, edge_index_s, edge_index_v), 1)
        edge_attr = torch.cat((torch.ones(edge_index_p.size(1)), torch.ones(edge_index_s.size(1))*2, torch.ones(edge_index_v.size(1))*3)).long()

        x = self.gat(x, edge_index, edge_attr)
        x = self.activation(x)
        x = self.projection(x)
        x = self.classifier(x)
        return x


class GAT_BatchNormalitzation(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25):
        super().__init__()
        self.variation = 'GAT_SelfAttention'

        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)

        self.attention = SelfAttention(in_channels, out_channels, out_channels)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x = self.contrastive(data)
        x = self.classifier(x)
        return x


class GAT_SELU_Alphadrop(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25):
        super().__init__()
        self.variation = 'GAT_SELU_Alphadrop'

        self.activation = nn.SELU()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout,
                        v2=True)
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout,
                        v2=True)

        self.attention = SelfAttention(in_channels, out_channels, out_channels, dropout=dropout)


        self.classifier = nn.Sequential(
            nn.AlphaDropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x = self.contrastive(data)
        x = self.classifier(x)
        return x



class GIN_ReLU(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_GIN=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        self.gin_p = GIN(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        )
        self.gin_s = GIN(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        )
        self.gin_v = GIN(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        )

        self.attention = SelfAttention(in_channels, out_channels, out_channels)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gin_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gin_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gin_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x = self.contrastive(data)
        x = self.classifier(x)
        return x
    

class GIN_tanh(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_GIN=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        self.gin_p = GIN(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        act=nn.Tanh(),
                        norm=nn.BatchNorm1d(hidden_channels),
                        )
        self.gin_s = GIN(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        act=nn.Tanh(),
                        norm=nn.BatchNorm1d(hidden_channels),
                        )
        self.gin_v = GIN(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        act=nn.Tanh(),
                        norm=nn.BatchNorm1d(hidden_channels),
                        )

        self.attention = SelfAttention(in_channels, out_channels, out_channels)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gin_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gin_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gin_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x = self.contrastive(data)
        x = self.classifier(x)
        return x


class GAE_encoder(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_GIN=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        self.gin_p = GIN(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        )
        self.gin_s = GIN(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        )
        self.gin_v = GIN(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_GIN,
                        )

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gin_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gin_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gin_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        return (x_p, x_s, x_v)


class GAE_model(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_GIN=0):
        super().__init__()
        
        self.encoder = GAE_encoder(dropout, hidden_channels, out_channels, num_layers, in_channels, dropout_GIN)

        self.z1_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))
        
        self.z2_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))
        
        self.z3_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))

        self.GAE = GAE(encoder=self.encoder)

        self.attention = SelfAttention2(in_channels, out_channels, out_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )
    
    def contrastive(self, data):
        x1, x2, x3 = self.GAE.encode(data)
        
        z1 = self.z1_proj(x1)
        z2 = self.z2_proj(x2)
        z3 = self.z3_proj(x3)
        return (z1, z2, z3)
    
    def compute_loss(self, z, data, mask=None):
        z1, z2, z3 = z 

        if mask is not None:
            # Set to zero the z embeddings of the nodes that are not in mask
            inverted_mask = torch.ones(data.x.size(0), dtype=bool)
            inverted_mask[mask] = False
            z1[inverted_mask], z2[inverted_mask], z3[inverted_mask] = 0, 0, 0

            # Remove the edges that connect to nodes in the mask
            mask_pos_edge1 = torch.ones(data.edge_index_p.size(1), dtype=bool)
            mask_pos_edge1[mask[data.edge_index_p[0]]] = False
            mask_pos_edge1[mask[data.edge_index_p[1]]] = False

            mask_pos_edge2 = torch.ones(data.edge_index_s.size(1), dtype=bool)
            mask_pos_edge2[mask[data.edge_index_s[0]]] = False
            mask_pos_edge2[mask[data.edge_index_s[1]]] = False

            mask_pos_edge3 = torch.ones(data.edge_index_v.size(1), dtype=bool)
            mask_pos_edge3[mask[data.edge_index_v[0]]] = False
            mask_pos_edge3[mask[data.edge_index_v[1]]] = False
            
            pos_edge1, pos_edge2, pos_edge3 = data.edge_index_p[:, mask_pos_edge1], data.edge_index_s[:, mask_pos_edge2], data.edge_index_v[:, mask_pos_edge3]
        
        else:
            pos_edge1, pos_edge2, pos_edge3 = data.edge_index_p, data.edge_index_s, data.edge_index_v

        if "neg_edge_index_p" in data and data.count < 20:
            neg_edge1, neg_edge2, neg_edge3 = data.neg_edge_index_p, data.neg_edge_index_s, data.neg_edge_index_v
            data.count += 1
        else:
            neg_edge1 = negative_sampling(pos_edge1, z1.size(0), method="dense")
            neg_edge2 = negative_sampling(pos_edge2, z2.size(0), method="dense")
            neg_edge3 = negative_sampling(pos_edge3, z3.size(0), method="dense")

            data.neg_edge_index_p = neg_edge1
            data.neg_edge_index_s = neg_edge2
            data.neg_edge_index_v = neg_edge3
            data.count = 0
    
        return self.GAE.recon_loss(z1, pos_edge1, neg_edge1) + self.GAE.recon_loss(z2, pos_edge2, neg_edge2) + self.GAE.recon_loss(z3, pos_edge3, neg_edge3)

    def encode(self, data):
        x1, x2, x3 = self.GAE.encode(data)
        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), 1)
        # Apply attention mechanism
        x = self.attention(x_embed, data.x)
        return x
    
    def forward(self, data):
        x = self.encode(data)
        x = self.classifier(x)
        return x
    


class GAE_encoder_GAT(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_GIN=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)

        self.gat_p = GAT(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_s = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)
        self.gat_v = GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        v2=True)

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gat_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gat_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gat_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        return (x_p, x_s, x_v)
    

class GAE_model_GAT(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_GAT=0):
        super().__init__()
        
        self.encoder = GAE_encoder_GAT(dropout, hidden_channels, out_channels, num_layers, in_channels, dropout_GAT)

        self.z1_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))
        
        self.z2_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))
        
        self.z3_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))

        self.GAE = GAE(encoder=self.encoder)

        self.attention = SelfAttention(in_channels, out_channels, out_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )
    
    def contrastive(self, data):
        x1, x2, x3 = self.GAE.encode(data)
        
        z1 = self.z1_proj(x1)
        z2 = self.z2_proj(x2)
        z3 = self.z3_proj(x3)
        return (z1, z2, z3)
    
    def compute_loss(self, z, data, mask=None):
        z1, z2, z3 = z 

        if mask is not None:
            # Set to zero the z embeddings of the nodes that are not in mask
            inverted_mask = torch.ones(data.x.size(0), dtype=bool)
            inverted_mask[mask] = False
            z1[inverted_mask], z2[inverted_mask], z3[inverted_mask] = 0, 0, 0

            # Remove the edges that connect to nodes in the mask
            mask_pos_edge1 = torch.ones(data.edge_index_p.size(1), dtype=bool)
            mask_pos_edge1[mask[data.edge_index_p[0]]] = False
            mask_pos_edge1[mask[data.edge_index_p[1]]] = False

            mask_pos_edge2 = torch.ones(data.edge_index_s.size(1), dtype=bool)
            mask_pos_edge2[mask[data.edge_index_s[0]]] = False
            mask_pos_edge2[mask[data.edge_index_s[1]]] = False

            mask_pos_edge3 = torch.ones(data.edge_index_v.size(1), dtype=bool)
            mask_pos_edge3[mask[data.edge_index_v[0]]] = False
            mask_pos_edge3[mask[data.edge_index_v[1]]] = False
            
            pos_edge1, pos_edge2, pos_edge3 = data.edge_index_p[:, mask_pos_edge1], data.edge_index_s[:, mask_pos_edge2], data.edge_index_v[:, mask_pos_edge3]
        
        else:
            pos_edge1, pos_edge2, pos_edge3 = data.edge_index_p, data.edge_index_s, data.edge_index_v

        if "neg_edge_index_p" in data and data.count < 20:
            neg_edge1, neg_edge2, neg_edge3 = data.neg_edge_index_p, data.neg_edge_index_s, data.neg_edge_index_v
            data.count += 1
        else:
            neg_edge1 = negative_sampling(pos_edge1, z1.size(0), method="dense")
            neg_edge2 = negative_sampling(pos_edge2, z2.size(0), method="dense")
            neg_edge3 = negative_sampling(pos_edge3, z3.size(0), method="dense")

            data.neg_edge_index_p = neg_edge1
            data.neg_edge_index_s = neg_edge2
            data.neg_edge_index_v = neg_edge3
            data.count = 0
    
        return self.GAE.recon_loss(z1, pos_edge1, neg_edge1) + self.GAE.recon_loss(z2, pos_edge2, neg_edge2) + self.GAE.recon_loss(z3, pos_edge3, neg_edge3)

    def encode(self, data):
        x1, x2, x3 = self.GAE.encode(data)
        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), 1)
        # Apply attention mechanism
        x = self.attention(x_embed, data.x)
        return x
    
    def forward(self, data):
        x = self.encode(data)
        x = self.classifier(x)
        return x
    

class GraphSAGE_model(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_SAGE=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        self.gin_p = GraphSAGE(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_SAGE,
                        )
        self.gin_s = GraphSAGE(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_SAGE,
                        )
        
        self.gin_v = GraphSAGE(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_SAGE,
                        )

        self.attention = SelfAttention(in_channels, out_channels, out_channels)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gin_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gin_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gin_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x = self.contrastive(data)
        x = self.classifier(x)
        return x
    

class PNA_model_2(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_PNA=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        aggregators = ['mean', 'mean', 'std', 'max']
        scalers = ['amplification', 'amplification', 'amplification', 'identity']
        deg = torch.tensor([1, 1, 1, 1])

        self.PNA_p = PNA(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        self.PNA_s = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        
        self.PNA_v = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )

        self.attention = SelfAttention(in_channels, out_channels, out_channels)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.PNA_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.PNA_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.PNA_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)

        return x

    def forward(self, data):
        x = self.contrastive(data)
        x = self.classifier(x)
        return x
    

class PNA_model_self_supervised(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_PNA=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)


        aggregators = ['mean', 'mean', 'mean']
        scalers = ['identity', 'identity', 'identity']
        deg = torch.tensor([2, 2, 2])

        self.gin_p = PNA(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        self.gin_s = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        
        self.gin_v = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )

        self.attention = SelfAttention2(in_channels, out_channels, out_channels)

        self.projection = nn.Sequential(nn.Tanh(),
                                        nn.Dropout(dropout),
                                        nn.Linear(out_channels, in_channels))


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )

    # Function to load the weights of the model (for perviouly trained models)
    def load_state_dict(self, state_dict):
        if 'attention.scoringDot.fc_q.weight' not in state_dict:
            state_dict['attention.scoringDot.fc_q.weight'] = state_dict['attention.fc_q.weight']
            state_dict['attention.scoringDot.fc_q.bias'] = state_dict['attention.fc_q.bias']

            state_dict['attention.scoringDot.fc_k.weight'] = state_dict['attention.fc_k.weight']
            state_dict['attention.scoringDot.fc_k.bias'] = state_dict['attention.fc_k.bias']

            state_dict['attention.scoringDot.fc_v.weight'] = state_dict['attention.fc_v.weight']
            state_dict['attention.scoringDot.fc_v.bias'] = state_dict['attention.fc_v.bias']
            super().load_state_dict(state_dict)

        else:
            super().load_state_dict(state_dict)

    def encode(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.gin_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.gin_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.gin_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x_p.unsqueeze(1), x_s.unsqueeze(1), x_v.unsqueeze(1)), 1)

        # Apply attention mechanism
        x = self.attention(x_embed, x)
        return x

    def contrastive(self, data):
        x = self.encode(data)
        x = self.projection(x)
        return x

    def forward(self, data):
        x = self.encode(data)
        x = self.classifier(x)
        return x


class GAE_encoder_PNA(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_PNA=0):
        super().__init__()
        self.activation = nn.Tanh()
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.p_norm = nn.BatchNorm1d(out_channels)
        self.s_norm = nn.BatchNorm1d(out_channels)
        self.v_norm = nn.BatchNorm1d(out_channels)

        aggregators = ['mean', 'mean', 'std', 'max']
        scalers = ['amplification', 'amplification', 'amplification', 'identity']
        deg = torch.tensor([1, 1, 1, 1])

        self.PNA_p = PNA(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        self.PNA_s = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )
        
        self.PNA_v = PNA(in_channels=in_channels,
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        )

    def forward(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        x_p = self.PNA_p(x, edge_index_p)
        x_p = self.p_norm(x_p)
        x_p = self.activation(x_p)

        x_s = self.PNA_s(x, edge_index_s)
        x_s = self.s_norm(x_s)
        x_s = self.activation(x_s)

        x_v = self.PNA_v(x, edge_index_v)
        x_v = self.v_norm(x_v)
        x_v = self.activation(x_v)

        return (x_p, x_s, x_v)
    

class GAE_model_PNA(torch.nn.Module):
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1, in_channels=25, dropout_PNA=0):
        super().__init__()
        
        self.encoder = GAE_encoder_PNA(dropout, hidden_channels, out_channels, num_layers, in_channels, dropout_PNA)

        self.z1_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))
        
        self.z2_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))
        
        self.z3_proj = nn.Sequential(nn.Linear(out_channels, out_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.Tanh(),
                                     nn.Linear(out_channels, out_channels))

        self.GAE = GAE(encoder=self.encoder)

        self.attention = SelfAttention2(in_channels, out_channels, out_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, int(out_channels*2/3)),
            nn.BatchNorm1d(int(out_channels*2/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*2/3), int(out_channels*1/3)),
            nn.BatchNorm1d(int(out_channels*1/3)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(out_channels*1/3), 2),
        )
    
    def contrastive(self, data):
        x1, x2, x3 = self.GAE.encode(data)
        
        z1 = self.z1_proj(x1)
        z2 = self.z2_proj(x2)
        z3 = self.z3_proj(x3)
        return (z1, z2, z3)
    
    def compute_loss(self, z, data, mask=None):
        z1, z2, z3 = z 

        if mask is not None:
            # Set to zero the z embeddings of the nodes that are not in mask
            inverted_mask = torch.ones(data.x.size(0), dtype=bool)
            inverted_mask[mask] = False
            z1[inverted_mask], z2[inverted_mask], z3[inverted_mask] = 0, 0, 0

            # Remove the edges that connect to nodes in the mask
            mask_pos_edge1 = torch.ones(data.edge_index_p.size(1), dtype=bool)
            mask_pos_edge1[mask[data.edge_index_p[0]]] = False
            mask_pos_edge1[mask[data.edge_index_p[1]]] = False

            mask_pos_edge2 = torch.ones(data.edge_index_s.size(1), dtype=bool)
            mask_pos_edge2[mask[data.edge_index_s[0]]] = False
            mask_pos_edge2[mask[data.edge_index_s[1]]] = False

            mask_pos_edge3 = torch.ones(data.edge_index_v.size(1), dtype=bool)
            mask_pos_edge3[mask[data.edge_index_v[0]]] = False
            mask_pos_edge3[mask[data.edge_index_v[1]]] = False
            
            pos_edge1, pos_edge2, pos_edge3 = data.edge_index_p[:, mask_pos_edge1], data.edge_index_s[:, mask_pos_edge2], data.edge_index_v[:, mask_pos_edge3]
        
        else:
            pos_edge1, pos_edge2, pos_edge3 = data.edge_index_p, data.edge_index_s, data.edge_index_v

        if "neg_edge_index_p" in data and data.count < 20:
            neg_edge1, neg_edge2, neg_edge3 = data.neg_edge_index_p, data.neg_edge_index_s, data.neg_edge_index_v
            data.count += 1
        else:
            neg_edge1 = negative_sampling(pos_edge1, z1.size(0), method="dense")
            neg_edge2 = negative_sampling(pos_edge2, z2.size(0), method="dense")
            neg_edge3 = negative_sampling(pos_edge3, z3.size(0), method="dense")

            data.neg_edge_index_p = neg_edge1
            data.neg_edge_index_s = neg_edge2
            data.neg_edge_index_v = neg_edge3
            data.count = 0
    
        return self.GAE.recon_loss(z1, pos_edge1, neg_edge1) + self.GAE.recon_loss(z2, pos_edge2, neg_edge2) + self.GAE.recon_loss(z3, pos_edge3, neg_edge3)

    def encode(self, data):
        x1, x2, x3 = self.GAE.encode(data)
        # Concate the embeddings (batch, 3, out_channels)
        x_embed = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), 1)
        # Apply attention mechanism
        x = self.attention(x_embed, data.x)
        return x
    
    def forward(self, data):
        x = self.encode(data)
        x = self.classifier(x)
        return x


class PNA_Edge_feat(torch.nn.Module):
    def __init__(self, dropout_PNA=0, dropout=0, hidden_channels=10, out_channels_GAT=5,  out_channels_proj=5, num_layers=1, in_channels=25):
        super().__init__()
        aggregators = ['mean', 'mean', 'mean']
        scalers = ['identity', 'identity', 'identity']
        deg = torch.tensor([2, 2, 2])
        
        self.in_norm = nn.BatchNorm1d(in_channels)
        self.PNA_norm = nn.BatchNorm1d(out_channels_GAT)

        # TanH activation
        self.activation = nn.Tanh()

        self.gat = PNA(in_channels=in_channels, 
                        hidden_channels=hidden_channels, 
                        num_layers=num_layers,
                        out_channels=out_channels_GAT,
                        dropout=dropout_PNA,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg)
        
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels_GAT, out_channels_proj),
            nn.BatchNorm1d(out_channels_proj),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(out_channels_proj, out_channels_proj),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels_proj, int((out_channels_proj+2)/2)),
            nn.BatchNorm1d(int((out_channels_proj+2)/2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int((out_channels_proj+2)/2), 2),
        )

    def contrastive(self, data):
        x, edge_index_p, edge_index_s, edge_index_v = data.x, data.edge_index_p, data.edge_index_s, data.edge_index_v
        x = self.in_norm(x)

        edge_index = torch.cat((edge_index_p, edge_index_s, edge_index_v), 1)
        edge_attr = torch.cat((torch.ones(edge_index_p.size(1)), torch.ones(edge_index_s.size(1))*2, torch.ones(edge_index_v.size(1))*3)).long()

        x = self.gat(x, edge_index, edge_attr)
        x = self.PNA_norm(x)
        x = self.activation(x)
        x = self.projection(x)
        return x

    def forward(self, data):
        x = self.contrastive(data)
        x = self.classifier(x)
        return x
