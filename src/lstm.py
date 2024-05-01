import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=25, num_layers=2, batch_first=True)
        self.linear = nn.Linear(25, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x