import pickle as pkl
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import roc_curve
from openTSNE import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import os
import sys
import yaml

# Runs to take into account:
# Run 17 Yelp Supervised
# Run 23 Yelp Supervised !!! (Best one) !!!
# Run 8 2 message Amazon/Yelp Supervised

if len(sys.argv) != 3:
    raise ValueError('The script needs two arguments: the name of the yaml file and the type of run.\nExample: python GMMClassifier.py name_yaml Supervised\n')
if sys.argv[2] not in ["Autoencoder", "SelfSupervisedContrastive", "Supervised", "SupervisedContrastive"]:
    raise ValueError(f'{sys.argv[2]} is not a valid run type. Use Autoencoder, SelfSupervisedContrastive, Supervised or SupervisedContrastive.')
# Get the name of the yaml file
name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Get the name of the type of run
run_type = sys.argv[2]
print(f'Run type: {run_type}')

# Open a yaml file with the parameters
with open(f'./Setups/{run_type}/{name_yaml}.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# Check if data is Amazon or Yelp
if params["data"] == "amz":
    run_path = f"./Runs/{run_type}/Amazon/{name_yaml}"
    data_file = loadmat('./Data/Amazon.mat')
elif params["data"] == "yelp":
    run_path = f"./Runs/{run_type}/Yelp/{name_yaml}"
    data_file = loadmat('./Data/YelpChi.mat')
else:
    raise ValueError(f'{params["data"]} is not a valid dataset. Use amz or yelp.')

# Load the embeddings and the masks
if "Supervised" == run_type:
    with open(f'{run_path}/embeds_contr_sup_{name_yaml}.pkl', "rb") as f:
        embeds = pkl.load(f)
    with open(f'{run_path}/train_test_val_masks_{name_yaml}.pkl', "rb") as f:
        train_mask, val_mask, test_mask, train_mask_contrastive = pkl.load(f)
elif run_type in ("Autoencoder", "SelfSupervisedContrastive", "SupervisedContrastive"):
    with open(f'{run_path}/Pickles/embeds_contr_sup_{name_yaml}.pkl', "rb") as f:
        embeds = pkl.load(f)
    with open(f'{run_path}/Pickles/train_test_val_masks_{name_yaml}.pkl', "rb") as f:
        train_mask, val_mask, test_mask, train_mask_contrastive = pkl.load(f)

# Create a folder for the GMM plots, results and prediction
if not os.path.exists(f'{run_path}/GMM'):
    os.makedirs(f'{run_path}/GMM')
if not os.path.exists(f'{run_path}/GMM/Plots'):
    os.makedirs(f'{run_path}/GMM/Plots')
if not os.path.exists(f'{run_path}/GMM/Results'):
    os.makedirs(f'{run_path}/GMM/Results')
if not os.path.exists(f'{run_path}/GMM/Predictions'):
    os.makedirs(f'{run_path}/GMM/Predictions')


labels = data_file['label'].flatten()

training_data = embeds[train_mask]
val_data = embeds[val_mask]
training_labels = labels[train_mask]
val_labels = labels[val_mask]

# Concatenate the training and validation data
training_data = np.concatenate([training_data, val_data], axis=0)
training_labels = np.concatenate([training_labels, val_labels], axis=0)


### DIMENSIONALITY REDUCTION ###

# TSNE
# tsne = TSNE(n_components=2, random_state=42, n_jobs=6, verbose=True)
# tsne = tsne.fit(training_data)
# training_data = tsne.transform(training_data)

# PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, 
#           random_state=42)
# training_data = pca.fit_transform(training_data)

# LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=1)
# training_data = lda.fit_transform(training_data, training_labels)

# Random Projection
# from sklearn.random_projection import GaussianRandomProjection
# rp = GaussianRandomProjection(n_components=2, random_state=42)
# training_data = rp.fit_transform(training_data)

# SVD
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=2, random_state=42)
# training_data = svd.fit_transform(training_data)

# Autoencoder (MLP regressor)
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(1, ), 
                   solver='adam',
                   activation='identity',
                   random_state=42)

# import torch
# np.random.seed(42)
# drop = torch.nn.Dropout(p=0.4)
# # Training cilcle
# for i in range(2):
#     # Shuffle the data
#     idx = np.random.permutation(len(training_data))
#     training_data_epoch = training_data[idx].copy() 
#     mlp.partial_fit(drop(torch.tensor(training_data_epoch)).numpy(), training_data_epoch)

# mlp.fit(training_data[training_labels == 0], training_data[training_labels == 0])
mlp.fit(training_data, training_data)

mlp_hidden = mlp.coefs_[0]
training_data = np.dot(training_data, mlp_hidden)
# Plot the loss curve
plt.plot(range(1, len(mlp.loss_curve_)+1), mlp.loss_curve_)
plt.savefig("MLP_Loss.png")
plt.close()


# Plot the features after dimensionality reduction
if training_data.shape[1] >= 2:
    # Plot the t-SNE of the training data
    plt.scatter(training_data[training_labels == 0][:, 0], training_data[training_labels == 0][:, 1], label='Normal')
    plt.scatter(training_data[training_labels == 1][:, 0], training_data[training_labels == 1][:, 1], label='Anomaly')
    plt.legend()
    plt.savefig(f"{run_path}/GMM/Plots/dimensionality_reduction.png")
    plt.close()
elif training_data.shape[1] == 1:
    plt.hist(training_data[training_labels == 1], bins=50, alpha=0.5, label='Anomaly')
    plt.hist(training_data[training_labels == 0], bins=50, alpha=0.5, label='Normal')
    plt.legend()
    plt.savefig(f"{run_path}/GMM/Plots/dimensionality_reduction.png")
    plt.close()
    

# Train GMM with embeddings of the non anomal class
gmm = GaussianMixture(n_components=1, init_params="k-means++" ,random_state=42)
gmm.fit(training_data[training_labels == 0])

# Plot the GMM density function of the normal class if the data is 1D
if training_data.shape[1] == 1:
    x = np.linspace(training_data.min(), training_data.max(), 1000)
    y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    plt.plot(x, y, label="GMM density function")
    plt.hist(training_data[training_labels == 1], bins=50, alpha=0.5, label='Anomaly', density=True)
    plt.hist(training_data[training_labels == 0], bins=50, alpha=0.5, label='Normal', density=True)
    plt.legend()
    plt.savefig(f"{run_path}/GMM/Plots/dimensionality_reduction_and_GMM.png")
    plt.close() 



# Compute the probability of each embedding to be in the GMM of the normal class
probs = gmm.score_samples(training_data)
# Convert the log probability to a probability between 0 and 1
probs = 1 - (1 / (1 + np.exp(probs)))

# Ploting the distribution of the GMM for each class
plt.hist(probs[training_labels == 1], bins=50, alpha=0.5, label='Anomaly')
plt.hist(probs[training_labels == 0], bins=50, alpha=0.5, label='Normal')
plt.legend()
plt.savefig(f"{run_path}/GMM/Plots/GMM_Distribution.png")
plt.close()


from sklearn.metrics import roc_curve

# fpr, tpr, thresholds = roc_curve(training_labels, training_data)
fpr, tpr, thresholds = roc_curve(training_labels, -probs)
# Precision Recall curve
from sklearn.metrics import precision_recall_curve
# precision, recall, _ = precision_recall_curve(training_labels, training_data)
precision, recall, _ = precision_recall_curve(training_labels, -probs)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve",
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.savefig(f"{run_path}/GMM/Plots/ROC_Curve_GNN.png")
plt.close()


# Precision Recall curve
plt.figure()
lw = 2
plt.plot(
    recall,
    precision,
    color="darkorange",
    lw=lw,
    label="Precision Recall curve",
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall curve")
plt.legend(loc="lower right")
plt.savefig(f"{run_path}/GMM/Plots/Precision_Recall_Curve_GNN.png")
plt.close()

# K-folds cross validation
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=50)
dict_splits_results = {}

best_thresholds = []
# For each split, train the model and get the results on the validation set
for i, (train_index, test_index) in enumerate(kf.split(probs, y=training_labels)):
    # Get the train and validation data of this split
    X_train2, X_test2 = probs[train_index], probs[test_index]
    # X_train2, X_test2 = training_data[train_index], training_data[test_index]
    y_train2, y_test2 = training_labels[train_index], training_labels[test_index]

    fpr, tpr, thresholds = roc_curve(y_train2, -X_train2)
    # fpr, tpr, thresholds = roc_curve(y_train2, X_train2)

    # Find the best threshold (Nearest to the top left corner)
    best_threshold = -thresholds[np.argmin(np.sqrt(fpr ** 2 + (1 - tpr) ** 2))]

    best_thresholds.append(best_threshold)
    # Get the results on the validation set
    y_pred2 = X_test2 < best_threshold
    dict_splits_results[f"split_{i}_val"] = classification_report(y_test2, y_pred2, target_names=["Negative", "Positive"], output_dict=True)


# Make the mean of the results on the splits

print("Results k fold val")
precision_neg = []
precision_pos = []
recall_neg = []
recall_pos = []
f1_neg = []
f1_pos = []
for key in dict_splits_results.keys():
    precision_neg.append(dict_splits_results[key]["Negative"]["precision"])
    precision_pos.append(dict_splits_results[key]["Positive"]["precision"])
    recall_neg.append(dict_splits_results[key]["Negative"]["recall"])
    recall_pos.append(dict_splits_results[key]["Positive"]["recall"])
    f1_neg.append(dict_splits_results[key]["Negative"]["f1-score"])
    f1_pos.append(dict_splits_results[key]["Positive"]["f1-score"])

# Plot the boxplot of the diferent tresholds in the different splits
plt.figure(figsize=(4, 5))
plt.boxplot(best_thresholds, 
            showmeans=True, 
            meanline=True, 
            labels=["Treshold positive/negative"])
plt.ylabel("Treshold value")
plt.savefig(f"{run_path}/GMM/Plots/boxplot_tresholds.png")
plt.close()


print("Mean results k folds val:")
print("Normal")
print(f"Precision: {np.median(precision_neg):.3} +- {np.std(precision_neg):.3}")
print(f"Recall: {np.median(recall_neg):.3} +- {np.std(recall_neg):.3}")
print(f"F1: {np.median(f1_neg):.3} +- {np.std(f1_neg):.3}")
print("")
print("Anomaly")
print(f"Precision: {np.median(precision_pos):.3} +- {np.std(precision_pos):.3}")
print(f"Recall: {np.median(recall_pos):.3} +- {np.std(recall_pos):.3}")
print(f"F1: {np.median(f1_pos):.3} +- {np.std(f1_pos):.3}")

print("")
print(f"Mean treshold: {np.mean(best_thresholds)} +- {np.std(best_thresholds)}")

# We get the meadian treshold of all the splits, so if there are outliers, they affect less to the final treshold
best_threshold = np.median(best_thresholds)
print("")
print(f"Best treshold: {best_threshold}")
print("")

# Boxplot of the diferent metrics during the k-folds cross validation
plt.figure(figsize=(10, 5))
plt.boxplot((precision_neg, recall_neg, f1_neg, precision_pos, recall_pos, f1_pos), 
            showmeans=True, 
            meanline=True, 
            meanprops={'color': 'blue'},
            medianprops={'color': 'orange'},
            labels=["Precision Normal", "Recall Normal", "F1 Normal", "Precision Anomaly", "Recall Anomaly", "F1 Anomaly"])
plt.ylabel("Metric value")
plt.ylim(0, 1)
plt.savefig(f"{run_path}/GMM/Plots/boxplot_metrics.png")
plt.close()



# Compute the test metrics
# test_data = tsne.transform(embeds[test_mask])
# test_data = pca.transform(embeds[test_mask])
# test_data = lda.transform(embeds[test_mask])
# test_data = rp.transform(embeds[test_mask])
# test_data = svd.transform(embeds[test_mask])
test_data = np.dot(embeds[test_mask], mlp_hidden)

test_labels = labels[test_mask]
test_probs = gmm.score_samples(test_data)
test_probs = 1 - (1 / (1 + np.exp(test_probs)))
# test_preds = test_data > best_threshold
test_preds = test_probs < best_threshold


print(classification_report(test_labels, test_preds))

report2save = classification_report(test_labels, test_preds, output_dict=True)

# ROC-AUC
# roc_auc = roc_auc_score(test_labels, test_data)
roc_auc = roc_auc_score(test_labels, -test_probs)
print("ROC-AUC: ", roc_auc)
report2save["ROC-AUC"] = roc_auc

# ROC curve test
# fpr, tpr, thresholds = roc_curve(test_labels, test_data)
fpr, tpr, thresholds = roc_curve(test_labels, -test_probs)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve",
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic test data")
plt.legend(loc="lower right")
plt.savefig(f"{run_path}/GMM/Plots/ROC_Curve_TEST.png")
plt.close()

# Average Precision
# average_precision = average_precision_score(test_labels, test_data)
average_precision = average_precision_score(test_labels, -test_probs)
print("Average Precision: ", average_precision)
report2save["AveragePrecision"] = average_precision

with open(f"{run_path}/GMM/Results/GMM_results.txt", "w") as f:
    f.write(str(report2save))

# Save test predictions as an .npy
test_preds = np.array(test_preds, dtype=int)
np.save(f"{run_path}/GMM/Predictions/test_preds.npy", test_preds)