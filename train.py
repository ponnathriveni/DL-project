import torch
import pandas as pd

from graph import create_graph
from model import FraudGNN


# Load transaction data
df = pd.read_csv(
    "transactions.csv"
)

# Create graph
graph, accounts = create_graph()

print("Graph created")
print("Number of accounts:", len(accounts))
print("Number of transactions:", len(df))


# Create model
model = FraudGNN(
    graph.num_node_features
)


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# Example training
for epoch in range(100):

    optimizer.zero_grad()

    output = model(
        graph.x,
        graph.edge_index
    )

    # Simple training target
    target = torch.zeros(
        graph.num_nodes,
        1
    )

    loss = torch.mean(
        (output - target) ** 2
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            "Epoch:",
            epoch + 1,
            "Loss:",
            loss.item()
        )


# Save model
torch.save(
    model.state_dict(),
    "fraud_gnn.pth"
)

print("Model saved successfully")