import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=4, num_layers=2, batch_first=True)
        self.last_layers = nn.Sequential() 
        self.last_layers.add_module("L1", nn.Linear(4, 4))
        self.last_layers.add_module("Drop", nn.Dropout())
        self.last_layers.add_module("RelU", nn.ReLU())
        self.last_layers.add_module("Out", nn.Linear(4, 1))

    def forward(self, x):
        x, _ = self.lstm(x) # Currently ignoring history / hidden state
        x = self.last_layers(x)
        return x
