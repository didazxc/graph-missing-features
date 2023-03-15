from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = output_size if i == num_layers - 1 else hidden_size
            if i > 0:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(in_size, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, egdes):
        return self.layers(x)
