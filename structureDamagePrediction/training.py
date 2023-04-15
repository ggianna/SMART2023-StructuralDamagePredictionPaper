import numpy as np
import torch.optim as optim
import torch.utils.data as data
import  torch.nn as nn, torch
from structureDamagePrediction.models import LSTMModel
from structureDamagePrediction.utils import StartEndLogger

class Trainer():
    def __init__(self, model = LSTMModel(), optimizer = None, loss_fn = nn.MSELoss(), n_epochs = 2000, validation_every = 10) -> None:
        self.model = model
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters())
        else:
            self.optimizer = optimizer        
        self.loss_fn = loss_fn        
        self.n_epochs = n_epochs
        self.validation_every = validation_every

    def train(self, train_loader):
        l = StartEndLogger()

        l.start("Training...")
        for epoch in range(self.n_epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                y_pred = self.model(X_batch)
                # Log result so far
                loss = self.loss_fn(y_pred, y_batch)
                l.log("Epoch: %d; Loss: %4.2f"%(epoch, loss.detach()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        l.end()


    def get_model(self):
        return self.model
    
    def get_loss_fn(self):
        return self.loss_fn