import os
import pandas as pd
import matplotlib.pyplot as plt


SELECTED_TASK = "MiraBest"
exps_dir = f"exps/{SELECTED_TASK}"


best_acc_list = []

# Function to find the best epoch based on early stopping criteria
def find_best_epoch(df, patience):
    best_epoch = 1
    best_loss = float('inf')
    wait = 0

    for epoch, row in df.iterrows():
        current_loss = row['test_loss']
        current_acc = row['test_acc']
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break
    return best_epoch

# Iterate over each folder
for folder in list(os.listdir(exps_dir)):
    print(folder)


    num_seeds = 10

    dfs = []
    accuracies = []
    n_seeds = 0

    for seed in range(num_seeds):
        # Construct the path to the seed folder
        seed_folder = os.path.join(exps_dir, folder, "seed_" + str(seed))

        loss_file = os.path.join(seed_folder, "losses.txt")
        try:
            df_of_the_current_seed = pd.read_csv(loss_file)
            dfs.append(df_of_the_current_seed)
            n_seeds += 1

            if "RndMul" in folder:
                patience = 100
            else:
                patience = 10

            best_epoch = find_best_epoch(df_of_the_current_seed, patience)
            if best_epoch is not None:
                accuracies.append(df_of_the_current_seed.loc[best_epoch, 'test_acc'])
            else:
                accuracies.append(0)

        except:
            continue


    if len(dfs) == 0:
        continue
    else:
        all_seeds_df = pd.concat(dfs, axis=0)


    name_list_accuracies = folder.split("_")[0] + "_accuracies.csv"


    average_df = all_seeds_df.groupby(all_seeds_df.index).mean()
    std_dev_df = all_seeds_df.groupby(all_seeds_df.index).std()

    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
    std_acc = (sum([(x - avg_acc) ** 2 for x in accuracies]) / len(accuracies)) ** 0.5 if accuracies else 0

    best_acc_list.append((f"{folder}",  n_seeds, avg_acc, std_acc))


sorted_best_acc_list = sorted(best_acc_list, key=lambda x: x[0], reverse=False)

# Print the sorted list
print("\nPrinting results per model \n")
print(f"{'Model_Encoding':<35} {'Num_Seeds':<10} {'Average_Accuracy':<20} {'Std_Accuracy':<20}\n")

print("-" * 98)
for entry in sorted_best_acc_list:
    model_encoding, num_seeds, avg_acc, std_acc = entry
    print(f"{model_encoding:<35} {num_seeds:<10} {avg_acc:<20.4f} {std_acc:<20.4f}")


sorted_df = pd.DataFrame(sorted_best_acc_list, columns=["Model_Encoding", "Num_Seeds", "Average_Accuracy", "Std_Accuracy"])
sorted_df.to_csv("sorted_best_test_accuracies.csv", index=False)

# Model encoding names are "integrated-k_value-encoding_type" or "rotational-k_value-n_value" or "classical"
# Please, build a new column with "integrated-encoding_type" or "rotational-n_value" or "classical" and another one with k_value (set it to 0 for classical)


# Extract the model encoding type and k_value from the folder name
sorted_df["k_value"] = sorted_df["Model_Encoding"].apply(lambda x: x.split("-")[1][1:] if x.startswith("integrated") or x.startswith("rot") else "2")
#sorted_df["model_name"] = sorted_df["Model_Encoding"].apply(lambda x: "integrated-" + x.split("-")[2].split(".")[0] if x.startswith("integrated") else "rotational" if x.startswith("rot") else "classical")

sorted_df["model_name"] = sorted_df["Model_Encoding"]
#print(sorted_df)

# Construct the new dataframe "data"
data = sorted_df.pivot(index="model_name", columns="k_value", values="Average_Accuracy")

#add to "data" the standard deviation of the true accuracies
std_data = sorted_df.pivot(index="model_name", columns="k_value", values="Std_Accuracy")
data_std = std_data
data_std.columns = [f"{col}_std" for col in data_std.columns]
data = pd.concat([data, data_std], axis=1)


if not os.path.exists("results"):
    os.makedirs("results")
data.to_csv(f"results/{SELECTED_TASK}.csv", index=True)
