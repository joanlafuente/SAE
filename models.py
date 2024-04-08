import torch
from torch import nn
from torch_geometric.nn import GCNConv, GAT

class GCN(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.activation = nn.LeakyReLU(0.1)

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

        self.projection = nn.Sequential(
            nn.Linear(150, 150),
        )

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
    def __init__(self, dropout=0, hidden_channels=10, out_channels=5, num_layers=1):
        super().__init__()
        self.variation = 'GCN_Att'
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

        self.attention = SelfAttention(25, out_channels, out_channels)

        self.projection = nn.Sequential(
            nn.Linear(25, out_channels),
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
    

class SelfAttention(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super().__init__()
    
        self.fc_q = nn.Linear(input_size, hidden_size) 
        self.fc_k = nn.Linear(embed_size, hidden_size) 
        self.fc_v = nn.Linear(embed_size, hidden_size) 

    def scoringDot(self, keys, query):
        # query: batch, input_size
        # keys: batch, 3, embed_size
        query = torch.tanh(self.fc_q(query)) # batch, hidden_size
        keys = torch.tanh(self.fc_k(keys))  # batch, 3, hidden_size

        query = query.unsqueeze(2) # batch, hidden_size, 1
        result = torch.bmm(keys, query) # batch, 3, 1
        result = result.squeeze(2)
        weights = result.softmax(1) # batch, 3
        return weights.unsqueeze(1) # batch, 1, 3  This represents the weights of each type of edge

    def forward(self, embeds, node_feat):
        # embeds: batch, 3, embed_size
        # node_feat: batch, input_size

        values = torch.tanh(self.fc_v(embeds)) # Batch, 3, hidden_size
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