from CustomDataset import load_custom_dataset

# To use the LArTPC dataset, ensure having the QCNN_datashare folder with the data.
try:
    from QCNN_datashare.data_loader import load_multiclass
except:
    pass



def get_LArTPC_full(batch_size = 16, train_ratio = 0.8, downscale = False, autocrop = True):
    try:
        dataset_train = load_multiclass(downscale = downscale, autocrop = autocrop)
    except:
        raise Exception("The dataset could not be loaded. Please check utils\get_dataset\dataset_LArTPC file.")
  
    X, y = dataset_train
    X = X.unsqueeze(1)
    info = "LArTPC dataset for multiclass classification. All classes loaded.\n"
    train_loader, test_loader = load_custom_dataset(batch_size, X, y, train_ratio)
    return train_loader, test_loader, info