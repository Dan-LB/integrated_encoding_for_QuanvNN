import os

import numpy as np
import torch
import torch.optim as optim


from quanvs.model_builder import CNN
from quanvs.Quanvolutional_Layer import QuanvolutionalLayer
from quanvs.model_builder import stack_quanv_on_top

from utils.read_config import load_config

from utils.get_dataset.dataset_Mirabest import get_MiraBest_binary
from utils.get_dataset.dataset_LArTPC import get_LArTPC_full

from utils.train_and_test import train, test


#Set to DEBUG_MODE = False to run the full experiment
DEBUG_MODE = False

#Select the task from ["MiraBest", "LArTPC"]
SELECTED_TASK = "MiraBest"


seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Constants used in this work
quantization = 50
n_quanv_channels = 8

epochs = 1000
lr = 0.0003

for seed in seeds:
    #set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)


    task_path = f"exps/{SELECTED_TASK}"
    if not os.path.exists(task_path):
        os.makedirs(task_path)

    # The results will be saved in a folder called "exps" in the root directory

    if SELECTED_TASK == "LArTPC":
        train_loader, test_loader, info = get_LArTPC_full(downscale = True, autocrop = False)
    elif SELECTED_TASK == "MiraBest":
        train_loader, test_loader, info = get_MiraBest_binary()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = "configs"

    # For each configuration file in the folder, the script will train a model
    for file in os.listdir(folder):
            config_name = file
            config_name = config_name.split(".")[0]

            file_path = f"{folder}/{config_name}"
            encoding, quanv_config, _, model_config = load_config(file_path)

            if SELECTED_TASK == "LArTPC":
                out_features = 7
            elif SELECTED_TASK == "MiraBest":
                out_features = 2

            model_config['fc2']['out_features'] = out_features # Set the number of output features

            model = CNN(device=device, config=model_config)
            model.to(device)

            print(f"Working on {config_name} for {SELECTED_TASK}")

            if not os.path.exists(f"{task_path}/{config_name}"):
                os.makedirs(f"{task_path}/{config_name}")

            if quanv_config is not None:
                # build the quanvolutional layer
                act = quanv_config['activation']
                kernel_size = quanv_config['kernel_size']
                padding = (float(kernel_size)-1)/2
                layer = QuanvolutionalLayer(1, n_quanv_channels, kernel_size=kernel_size, 
                                            stride=1, padding=padding,
                                            quantization=quantization,
                                            encoding_approach=encoding,
                                            encoding_config=quanv_config,
                                            )

                # stack the quanvolutional layer on top of the classical model
                model = stack_quanv_on_top(layer, model)
                model.to(device)

                if DEBUG_MODE:
                    print("Debugging mode... skipping preprocessing")
                else:
                    p_train_loader_ = model.quanv_preprocess(train_loader, verbose = False)
                    p_test_loader_ = model.quanv_preprocess(test_loader, verbose = False)
                    model.preprocessed = True
            
            else:
                p_train_loader_ = train_loader
                p_test_loader_ = test_loader


            optimizer = optim.Adam(model.parameters(), lr=lr)

            #save model structure
            with open(f"{task_path}/{config_name}/structure.txt", "w") as f:
                f.write(str(model))

            if encoding is not None: 
                with open(f"{task_path}/{config_name}/quanv_structure.txt", "w") as f:
                    f.write(str(layer))
                with open(f"{task_path}/{config_name}/quanv_encoding_config.txt", "w") as f:
                    f.write(str(layer.encoding_config))
                    f.write(str(layer.counters))
                    f.write(str(quanv_config))
                

            if not os.path.exists(f"{task_path}/{config_name}/seed_{seed}"):
                os.makedirs(f"{task_path}/{config_name}/seed_{seed}")

            data_lines = ["epoch,train_loss,train_acc,test_loss,test_acc\n"]

            if DEBUG_MODE:
                print("Debugging mode... skipping training...")
            else:
                for epoch in range(1, epochs + 1):
                    train_loss, train_acc = train(model, device, p_train_loader_, optimizer, epoch)
                    test_loss, test_acc = test(model, device, p_test_loader_)
                    # Store each line in the list instead of writing to the file directly
                    data_lines.append(f"{epoch},{train_loss},{train_acc},{test_loss},{test_acc}\n")

            # Write all collected data to the file at once
            with open(f"{task_path}/{config_name}/seed_{seed}/losses.txt", "w") as f:
                f.writelines(data_lines)
