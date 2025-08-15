import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
from model.tgnn_model import TGNN

def train():
    # Load and preprocess data
    df = pd.read_csv('data/synthetic_traffic.csv')

    # Convert 'timestamp' to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set timestamp as index (optional)
    df.set_index('timestamp', inplace=True)

    # Normalize features (excluding non-numeric columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # Prepare sequences for training
    sequence_length = 12
    features = df.values
    X, y = [], []

    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(features[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Model
    
    model = TGNN(input_size=X.shape[2], hidden_size=64, output_size=y.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_values = []
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        actuals = y_test.numpy()
        mse = np.mean((predictions - actuals) ** 2)

    # Create plot folder if not exists
    os.makedirs('static/images', exist_ok=True)

    # Plot loss
    plt.figure()
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    loss_plot_path = 'static/images/loss_plot.png'
    plt.savefig(loss_plot_path)

    # Predictions vs Actuals (first feature)
    plt.figure()
    plt.plot(predictions[:, 0], label='Predicted')
    plt.plot(actuals[:, 0], label='Actual')
    plt.title('Prediction vs Actual')
    plt.legend()
    pred_plot_path = 'static/images/predictions.png'
    plt.savefig(pred_plot_path)

    # MSE plot
    plt.figure()
    plt.bar(['MSE'], [mse])
    plt.title('Mean Squared Error')
    mse_plot_path = 'static/images/mse_plot.png'
    plt.savefig(mse_plot_path)

    # Dummy live simulation (copy a placeholder GIF to correct path if needed)
    gif_path = 'static/images/live_simulation.gif'

    return loss_values, predictions, actuals, [loss_plot_path, pred_plot_path, mse_plot_path, gif_path], mse
