import torch.nn as nn
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import sklearn.dummy as dummy
import sklearn.tree
from abc import ABC

class SKLearnModel(ABC):
    def fit(self, X, Y):
        pass

class LSTMRegressionModel(nn.Sequential):
    def __init__(self, device, input_size = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=3, 
                            # dropout=0.1, 
                            num_layers=1, batch_first=True, device=device)
        self.last_layers = nn.Sequential() 
        # self.last_layers.add_module("L1", nn.Linear(3, 15))
        # self.last_layers.add_module("L2", nn.Linear(15, 3))
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

class RNNRegressionModel(nn.Sequential):
    def __init__(self, device,input_size = 3):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=input_size, dropout=0.1, num_layers=3, batch_first=True, device=device)
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

# Regression
class MLPRegressor(nn.Sequential):
    def __init__(self, device, input_size = 3, num_classes = 3):
        super().__init__()
        self.last_layers = nn.Sequential() 
        self.last_layers.add_module("In", nn.Linear(input_size, input_size))
        # self.last_layers.add_module("RelU", nn.ReLU())
        self.last_layers.add_module("Out", nn.Linear(input_size, num_classes))
        self.last_layers.to(device=device)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.last_layers(x)
        return x

class LinearRegressor(nn.Sequential, SKLearnModel):
    def __init__(self, input_size = 3) -> None:
        nn.Sequential.__init__(self)
        self.identity = nn.Sequential()
        self.identity.add_module("Id", nn.Identity(input_size))
        self.linear = LinearRegression()

    def forward(self, x):
        x = self.identity(x)
        return torch.from_numpy(self.linear.predict(x.detach().cpu().numpy()))
    
    def fit(self, X, y):
        return self.linear.fit(X.detach().cpu().numpy(),y.detach().cpu().numpy())
    

# Classification
class LSTMClassificationModel(nn.Sequential):
    def __init__(self, device, input_size = 3, num_classes = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size, 
                            # dropout=0.1, 
                            num_layers=1, batch_first=True, device=device)
        self.last_layers = nn.Sequential() 
        # self.last_layers.add_module("L1", nn.Linear(3, 15))
        # self.last_layers.add_module("L2", nn.Linear(15, 3))
        self.last_layers.add_module("RelU", nn.ReLU())
        self.last_layers.add_module("Out", nn.Linear(3, num_classes))
        self.last_layers.to(device=device)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x, _ = self.lstm(x) # Currently ignoring history / hidden state
        x = x[:, -1, :] # Extract only last timestep
        x = self.last_layers(x)
        return x
    
# Classification
class MLPClassifier(nn.Sequential):
    def __init__(self, device, input_size = 3, num_classes = 3):
        super().__init__()
        self.last_layers = nn.Sequential() 
        # self.last_layers.add_module("L1", nn.Linear(3, 15))
        # self.last_layers.add_module("L2", nn.Linear(15, 3))
        self.last_layers.add_module("In", nn.Linear(input_size, input_size))
        # self.last_layers.add_module("RelU", nn.ReLU())
        self.last_layers.add_module("Out", nn.Linear(input_size, num_classes))
        self.last_layers.to(device=device)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.last_layers(x)
        return x


class KNNModel(nn.Sequential, SKLearnModel):
    def __init__(self, n_neighbors, input_size = 3, num_classes = 3) -> None:
        nn.Sequential.__init__(self)
        self.identity = nn.Sequential()
        self.identity.add_module("Id", nn.Identity(input_size))
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def forward(self, x):
        x = self.identity(x)
        return torch.from_numpy(self.knn.predict(x.detach().cpu().numpy()))
    
    def fit(self, X, y):
        return self.knn.fit(X.detach().cpu().numpy(),y.detach().cpu().numpy())

class DummyModel(nn.Sequential, SKLearnModel):
    def __init__(self, input_size = 3, num_classes = 3, strategy='stratified') -> None:
        nn.Sequential.__init__(self)
        self.identity = nn.Sequential()
        self.identity.add_module("Id", nn.Identity(input_size))
        self.dummy = dummy.DummyClassifier(strategy=strategy)

    def forward(self, x):
        x = self.identity(x)
        return torch.from_numpy(self.dummy.predict(x.detach().cpu().numpy()))
    
    def fit(self, X, y):
        return self.dummy.fit(X.detach().cpu().numpy(),y.detach().cpu().numpy())

class DecisionTreeModel(nn.Sequential, SKLearnModel):
    def __init__(self, input_size = 3, num_classes = 3) -> None:
        nn.Sequential.__init__(self)
        self.identity = nn.Sequential()
        self.identity.add_module("Id", nn.Identity(input_size))
        self.dtree = sklearn.tree.DecisionTreeClassifier()

    def forward(self, x):
        x = self.identity(x)
        return torch.from_numpy(self.dtree.predict(x.detach().cpu().numpy()))
    
    def fit(self, X, y):
        return self.dtree.fit(X.detach().cpu().numpy(),y.detach().cpu().numpy())
