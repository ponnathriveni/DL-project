import pandas as pd
import torch
from torch_geometric.data import Data
import pandas as pd


def create_graph():

    df = pd.read_csv("transactions.csv")

    accounts = list(
        set(df["sender"]) |
        set(df["receiver"])
    )

    print("Transactions:", len(df))
    print("Accounts:", len(accounts))

    return df, accounts

def create_graph():

    df = pd.read_csv("transactions.csv")

    accounts = list(
        set(df["sender"]) | set(df["receiver"])
    )

    account_to_id = {
        account: i
        for i, account in enumerate(accounts)
    }

    edges = []

    for _, row in df.iterrows():

        sender = account_to_id[row["sender"]]
        receiver = account_to_id[row["receiver"]]

        edges.append([sender, receiver])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    # Simple node feature
    x = torch.ones(
        (len(accounts), 1),
        dtype=torch.float
    )

    graph = Data(
        x=x,
        edge_index=edge_index
    )

    return graph, accounts