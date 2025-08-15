import pandas as pd
import torch

def create_graph_data(df):
    df = df.drop(columns=['timestamp'])
    X = df.values[:-1]
    y = df.values[1:]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y[:, 0], dtype=torch.float32).view(-1, 1)
    return X_tensor, y_tensor

def load_data(filepath='data/sample_traffic.csv'):
    df = pd.read_csv(filepath)
    X, y = create_graph_data(df)
    X = X.unsqueeze(1)  # Add batch dimension: (seq_len, 1, features)
    return X, y
