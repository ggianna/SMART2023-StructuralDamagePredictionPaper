import torch.nn as nn

class LSTMModel(nn.Sequential):
    def __init__(self, device):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=3, dropout=0.1, num_layers=3, batch_first=True, device=device)
        self.last_layers = nn.Sequential() 
        self.last_layers.add_module("L1", nn.Linear(3, 15))
        self.last_layers.add_module("L2", nn.Linear(15, 3))
        self.last_layers.add_module("RelU", nn.ReLU())
        self.last_layers.add_module("Out", nn.Linear(3, 1))
        self.last_layers.to(device=device)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x, _ = self.lstm(x) # Currently ignoring history / hidden state
        x = x[:, -1, :] # Extract only last timestep
        x = self.last_layers(x)
        return x

class RNNModel(nn.Sequential):
    def __init__(self, device):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=3, dropout=0.1, num_layers=3, batch_first=True, device=device)
        self.last_layers = nn.Sequential() 
        self.last_layers.add_module("L1", nn.Linear(3, 3))
        self.last_layers.add_module("L2", nn.Linear(3, 3))
        self.last_layers.add_module("RelU", nn.ReLU())
        self.last_layers.add_module("Out", nn.Linear(3, 1))
        self.last_layers.to(device=device)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x, _ = self.rnn(x) # Currently ignoring history / hidden state
        x = x[:, -1, :] # Extract only last timestep
        x = self.last_layers(x)
        return x
