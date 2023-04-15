from structureDamagePrediction.datahandling import StructuralDamageDataAndMetadataReader, StructuralDamageDataset
from datetime import datetime
from structureDamagePrediction.utils import StartEndLogger
import numpy as np
import structureDamagePrediction.models as models
from structureDamagePrediction.training import Trainer
from torch.utils.data import DataLoader
import torch

def main():
    # Init utils
    l = StartEndLogger()
    # Init reader
    reader = StructuralDamageDataAndMetadataReader()
    # Read data and metadata
    data, meta_data = reader.read_data_and_metadata()

    # Meta-data format
    # case_id, dmg_perc, dmg_tensor, dmg_loc_x, dmg_loc_y    
    dataset = StructuralDamageDataset(data, meta_data, 1)

    # Choose test instance index (leave one out)
    test_instance_idx=np.random.choice(len(dataset) - 1)
    l.log("Selected instance: %d"%(test_instance_idx))
    # Create train and test data
    train_data = []
    test_data = []
    for idx,entry in enumerate(dataset):
        if idx == test_instance_idx:
            test_data.append(entry)
        else:
            train_data.append(entry)
    
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l.log("Device for learning: %s"%(device.type))

    model = models.LSTMModel(device=device)
    
    trainer = Trainer(model, n_epochs=10, device=device)
    trainer.train(train_dataloader)
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
                test_rmse = np.sqrt(trainer.loss_fn(y_pred, y_test).cpu())
                l.log("Loss: %8.6f"%(test_rmse))
    l.end()



if __name__ == "__main__":
    main()