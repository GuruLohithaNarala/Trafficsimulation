import torch
import torch.nn as nn

class TGNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TGNN, self).__init__()
        self.hidden_size = hidden_size

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        # Final output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        out, _ = self.lstm(x)  # out shape: [batch_size, sequence_length, hidden_size]
        out = out[:, -1, :]    # Take the output of the last time step
        out = self.fc(out)     # Final output
        return out
