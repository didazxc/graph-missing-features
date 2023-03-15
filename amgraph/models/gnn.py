from torch import nn
from torch_geometric.nn import GATConv, SAGEConv, GCNConv


class GNN(nn.Module):
    
    def __init__(self, num_attrs, num_classes, num_layers=2, hidden_size=256, dropout=0.5, residual=True, conv='gcn'):
        super().__init__()
        self.residual = residual
        self.conv = conv
        self.relu = nn.ReLU()
        self.drop_prob = dropout
        self.dropout = nn.Dropout()

        layers = []
        for i in range(num_layers):
            in_size = num_attrs if i == 0 else hidden_size
            out_size = num_classes if i == num_layers - 1 else hidden_size
            layers.append(self.to_conv(in_size, out_size))
        self.layers = nn.ModuleList(layers)

    def to_conv(self, in_size, out_size, heads=8):
        if self.conv == 'gcn':
            return GCNConv(in_size, out_size)
        elif self.conv == 'sage':
            return SAGEConv(in_size, out_size)
        elif self.conv == 'gat':
            return GATConv(in_size, out_size // heads, heads=heads, dropout=self.drop_prob)
        else:
            raise ValueError(self.conv)

    def forward(self, x, edges):
        out = x
        for i, layer in enumerate(self.layers[:-1]):
            out2 = out
            if i > 0:
                out2 = self.dropout(out2)
            out2 = layer(out2, edges)
            out2 = self.relu(out2)
            if i > 0 and self.residual:
                out2 = out2 + out
            out = out2
        out = self.dropout(out)
        return self.layers[-1](out, edges)
