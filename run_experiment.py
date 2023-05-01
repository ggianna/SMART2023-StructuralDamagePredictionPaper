from structureDamagePrediction.datahandling import StructuralDamageDataAndMetadataReader, StructuralDamageDataset
from datetime import datetime
from structureDamagePrediction.utils import StartEndLogger
import numpy as np
import structureDamagePrediction.models as models
from structureDamagePrediction.training import Trainer
from torch.utils.data import DataLoader
import torch, math
from sklearn.model_selection import train_test_split
import scipy.stats as stats

def main():
    # Init utils
    l = StartEndLogger()
    # Init reader
    reader = StructuralDamageDataAndMetadataReader()
    # Read data and metadata
    data, meta_data = reader.read_data_and_metadata()

    # Transformation function for classification
    def transform_func(x):
        idx = [0.025, 0.05, 0.10].index(x)
        return idx

    # Regression (no change)    
    # transform_func = None

    # Meta-data format
    # case_id, dmg_perc, dmg_tensor, dmg_loc_x, dmg_loc_y    
    dataset = StructuralDamageDataset(data, meta_data, 
                                      tgt_tuple_index_in_metadata=1,  tgt_row_in_metadata=None, tgt_col_in_metadata=None, # What to use: dmg percentage
                                      transform_func=transform_func)
    number_of_runs = 3


    # Only one should be True
    leave_one_out = True # TODO Implement
    stratify = False


    predicted_list = []
    real_list = []

    for iRun in range(number_of_runs):
        l.log("+++++ Starting run #%d"%(iRun))

        if stratify:
            _, test_instance_idx = train_test_split(np.arange(len(dataset)),
                                                        test_size=0.20,
                                                        random_state=5, # Reproducibility
                                                        shuffle=True,
                                                        stratify=list(dataset.labels())
                                                        )
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

        # Regression
        # model = models.LSTMRegressionModel(device=device)
        # model = models.RNNModel(device=device)
        # loss_fn=torch.nn.L1Loss()

        # Classification
        model = models.LSTMClassificationModel(device=device, num_classes=3)
        loss_fn=torch.nn.CrossEntropyLoss()
        
        trainer = Trainer(model, 
                        optimizer=torch.optim.Adam(params=model.parameters(), 
                                                            betas=(0.9, 0.999), eps=10e-7, lr=1e-4) , 
                        # optimizer=torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.1),
                        n_epochs=500, device=device, loss_fn=loss_fn)
        trainer.train(train_dataloader,min_abs_loss_change=0.0001, patience_epochs=200, sufficient_loss=0.001, output_every=100)
        final_model = trainer.get_model()

        l.start("Validation...")
        # Validation every some steps
        final_model.eval()
        with torch.no_grad():
            for X_test, y_test in test_dataloader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                real_list.append(y_test.item())

                y_pred = final_model(X_test).detach()
                predicted_list.append(y_pred.item())
                
                test_loss = trainer.loss_fn(y_pred, y_test).cpu()
                prc_loss = 100 * test_loss / y_test
                l.log("True: %8.6f -- Predicted: %8.6f (Loss: %8.6f; Percantile: %5.2f%%)"%(y_test.cpu().item(), y_pred.cpu().item(), test_loss ,prc_loss))

        l.end()

    l.log("Outputting overall results list:")
    l.log("\n".join(map(lambda x: str(x),list(zip(real_list, predicted_list)))))
    corr, p = stats.spearmanr(real_list, predicted_list)
    l.log("Correlation: %f (p-val: %f)"%(corr, p))

if __name__ == "__main__":
    main()