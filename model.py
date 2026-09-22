import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class FraudGNN(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            16
        )

        self.conv2 = GCNConv(
            16,
            8
        )

        self.output = nn.Linear(
            8,
            1
        )

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.output(x)

        return x