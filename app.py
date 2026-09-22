import torch
from fastapi import FastAPI
from pydantic import BaseModel

from graph import create_graph
from model import FraudGNN


app = FastAPI()


class Transaction(BaseModel):
    sender: str
    receiver: str
    amount: float


@app.get("/")
def home():
    return {
        "message": "LedgerGraph API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    graph, accounts = create_graph()

    model = FraudGNN(
        graph.num_node_features
    )

    model.load_state_dict(
        torch.load(
            "fraud_gnn.pth",
            weights_only=True
        )
    )

    model.eval()

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index
        )

        probabilities = torch.sigmoid(output)

    return {
        "sender": transaction.sender,
        "receiver": transaction.receiver,
        "amount": transaction.amount,
        "message": "Prediction completed"
    }