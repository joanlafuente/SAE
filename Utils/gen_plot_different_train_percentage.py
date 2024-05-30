import sys


if len(sys.argv) != 3:
    raise ValueError('The script needs two arguments: the name of the yaml file and the type of run.\nExample: python gen_plot_different_train_percentage.py name_yaml Supervised\n')
if sys.argv[2] not in ["Autoencoder", "SelfSupervisedContrastive", "Supervised", "SupervisedContrastive"]:
    raise ValueError(f'{sys.argv[2]} is not a valid run type. Use Autoencoder, SelfSupervisedContrastive, Supervised or SupervisedContrastive.')
# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')
# Get the name of the type of run
run_type = sys.argv[2]
print(f'Run type: {run_type}')
# Obtaining the path of the run
run_path = f"./Runs/{run_type}/Yelp/{name_yaml}"


AP_list = []
ROC_list = []
for i in ["20%", "30%", "40%", "50%", "60%", ""]:
    if i == "":
        tmp_path = run_path
        tmp_name_yaml = name_yaml
    else:
        tmp_path = run_path + f"_{i}"
        tmp_name_yaml = name_yaml + f"_{i}"
    # Load the .txt with the results
    with open(f'{tmp_path}/Report/cls_{tmp_name_yaml}.txt', 'r') as file:
        results = file.read()

    # Convert the string to a dictionary
    results = eval(results)
    AP_list.append(results["AP"])
    ROC_list.append(results["ROC_AUC"])

print(AP_list)
print(ROC_list)

# Plot the AP and ROC_AUC
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(["20%", "30%", "40%", "50%", "60%", "70%"], AP_list, label='AP')
plt.plot(["20%", "30%", "40%", "50%", "60%", "70%"], ROC_list, label='AUC')
plt.xlabel('Percentage of training data')
plt.ylabel('Score')
plt.title(f'AP and AUC scores for different training data percentages')
plt.legend()
plt.ylim(0.65, 1)
plt.savefig(f'{run_path}/Plots/AP_ROC_AUC_training_percentages.png')