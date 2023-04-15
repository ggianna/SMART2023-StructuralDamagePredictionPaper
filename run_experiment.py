from structureDamagePrediction.datahandling import StructuralDamageDataAndMetadataReader, StructuralDamageDataset
from datetime import datetime
from structureDamagePrediction.utils import StartEndLogger
import numpy as np
import structureDamagePrediction.models as models
from structureDamagePrediction.training import Trainer
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

def main():
    # Init utils
    l = StartEndLogger()
    # Init reader
    reader = StructuralDamageDataAndMetadataReader()
    # Read data and metadata
    data, meta_data = reader.read_data_and_metadata()

    # Meta-data format
    # case_id, dmg_perc, dmg_tensor, dmg_loc_x, dmg_loc_y    
    dataset = StructuralDamageDataset(data, meta_data, 2, 0, 1)

    stratify = True
    if stratify:
        train_data_idx, test_instance_idx = train_test_split(np.arange(len(dataset)),
                                                    test_size=0.1,
                                                    random_state=999,
                                                    shuffle=True,
                                                    stratify=list(dataset.labels()))
    else:
        # Choose test instance indexes
        test_instance_idx=np.random.choice(list(range(0, len(dataset))),  size = int(len(dataset) / 5), replace=False)
        l.log("Selected instances: %s"%(str(test_instance_idx)))
        
    # Create train and test data
    train_data = []
    test_data = []
    for idx,entry in enumerate(dataset):
        if idx in test_instance_idx:
            test_data.append(entry)
        else:
            train_data.append(entry)

    l.log("Train / test sizes: %4d /%4d"%(len(train_data), len(test_data)))
    
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l.log("Device for learning: %s"%(device.type))

    #model = models.LSTMModel(device=device)
    model = models.RNNModel(device=device)
    
    trainer = Trainer(model, n_epochs=1000, device=device, loss_fn=torch.nn.MSELoss())
    trainer.train(train_dataloader,min_abs_loss_change=0.001, patience_epochs=500, sufficient_loss=0.05, output_every=100)
    final_model = trainer.get_model()

    l.start("Validation...")
    # Validation every some steps
    for X_test, y_test in test_dataloader:
        final_model.eval()
        with torch.no_grad():
            for X_test, y_test in test_dataloader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)

                y_pred = final_model(X_test).detach()
                test_rmse = trainer.loss_fn(y_pred, y_test).cpu()
                l.log("True: %8.6f -- Predicted: %8.6f (Loss: %8.6f)"%(y_test.cpu().item(), y_pred.cpu().item(), test_rmse))
    l.end()



if __name__ == "__main__":
    main()