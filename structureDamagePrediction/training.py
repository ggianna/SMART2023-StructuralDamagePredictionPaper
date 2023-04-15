import numpy as np
import torch.optim as optim
import torch.utils.data as data
import  torch.nn as nn, torch
from structureDamagePrediction.models import LSTMModel
from structureDamagePrediction.utils import StartEndLogger

class Trainer():
    def __init__(self, model, optimizer = None, loss_fn = nn.MSELoss(), n_epochs = 2000, device = None) -> None:
        self.model = model
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters())
        else:
            self.optimizer = optimizer        
        self.loss_fn = loss_fn        
        self.n_epochs = n_epochs
        self.device = device

    def train(self, train_loader):
        l = StartEndLogger()

        l.start("Training...")
        for epoch in range(self.n_epochs):
            epoch_total_loss = 0.0
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                # Update epoch total
                epoch_total_loss += loss.detach()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Log result so far
            l.log("Epoch: %d; Loss: %8.6f"%(epoch, epoch_total_loss))
            
        l.end("Training...")


    def get_model(self):
        return self.model
    
    def get_loss_fn(self):
        return self.loss_fn