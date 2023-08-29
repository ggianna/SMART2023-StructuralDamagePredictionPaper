import numpy as np
import torch.optim as optim
import torch.utils.data as data
import  torch.nn as nn, torch
from structureDamagePrediction.models import LSTMRegressionModel
from structureDamagePrediction.utils import StartEndLogger
from typing import Callable
import time

class NeuralNetTrainer():
    def __init__(self, model, optimizer = None, loss_fn = nn.L1Loss(), n_epochs = 2000, device = None) -> None:
        self.model = model
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters())
        else:
            self.optimizer = optimizer        
        self.loss_fn = loss_fn        
        self.n_epochs = n_epochs
        self.device = device

    def train(self, train_loader, patience_epochs = 3, min_abs_loss_change = 10e-2, sufficient_loss = 10e-4, output_every = 1, 
              label_encoder: Callable[[torch.Tensor],torch.Tensor] = None):
        l = StartEndLogger()

        l.start("Training...")
        self.epochs_below_min_change = 0 # Init patience
        self.min_loss = float('+inf') # Arbitrarily high
        self.last_loss = float('+inf')

        start_time = time.time()

        for epoch in range(self.n_epochs):
            epoch_total_loss = 0.0
            self.model.train()
            iBatchCnt = 0 # DEBUG
            for X_batch, y_batch in train_loader:
                iBatchCnt += 1
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)

                loss = self.loss_fn(y_pred, y_batch)
                if (torch.isnan(loss)):
                    l.log("WARNING!!! Epoch: %d; Batch: %d; Loss is NAN! Ignoring..."%(epoch, iBatchCnt))
                    self.optimizer.zero_grad()
                    continue
                else:
                    # DOES NOT WORK: Parameter clipping to avoid NaNs
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5, error_if_nonfinite=True)

                    # Update epoch total
                    epoch_total_loss += loss.detach()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            # Log result so far
            if epoch % output_every == 0:
                l.log("Epoch: %d; Loss: %8.6f (sec per epoch %6.2f)"%(epoch, epoch_total_loss, (time.time() - start_time) / (epoch + 1)))
            
            # If the epoch loss is worse than the minimum we have achieved, or there is no significant change
            if epoch_total_loss >= self.min_loss or (abs(epoch_total_loss - self.last_loss) < min_abs_loss_change) :
                self.epochs_below_min_change += 1 # Update patience
            else:
                # DEBUG INFO
                # if self.epochs_below_min_change > 0:
                #     l.log("Patience reset after %d epochs."%(self.epochs_below_min_change))
                self.epochs_below_min_change = 0 # Reset patience
            
            self.min_loss = min(epoch_total_loss, self.min_loss) # Update min loss
            self.last_loss = epoch_total_loss # Update last loss

            if self.epochs_below_min_change > patience_epochs:
                l.end("Training (due to patience exhaustion)...")
                return
            if  epoch_total_loss <= sufficient_loss:
                l.end("Training (due to sufficient loss in epoch)...")
                return

        l.end("Training...")


    def get_model(self):
        return self.model
    
    def get_loss_fn(self):
        return self.loss_fn