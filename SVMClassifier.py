import pickle as pkl
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import roc_curve
from openTSNE import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import os
import sys
import yaml


"""
Script to train a SVM classifier to detect anomalies on 
the embeddings of a previously trained model.

In this script there is the code to use different dimensionality reduction techniques,
but the best results on our experiments were obtained with the Autoencoder.

Interesting experiments to take into account:
    - Run 17 Yelp Supervised
    - Run 23 Yelp Supervised !!! (Best one) !!!
    - Run 8 2 message Amazon/Yelp Supervised
"""

# Check if the script has the correct input arguments
if len(sys.argv) != 3:
    raise ValueError('The script needs two arguments: the name of the yaml file and the type of run.\nExample: python SVMClassifier.py name_yaml Supervised\n')
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
if not os.path.exists(f'{run_path}/SVM'):
    os.makedirs(f'{run_path}/SVM')
if not os.path.exists(f'{run_path}/SVM/Plots'):
    os.makedirs(f'{run_path}/SVM/Plots')
if not os.path.exists(f'{run_path}/SVM/Results'):
    os.makedirs(f'{run_path}/SVM/Results')
if not os.path.exists(f'{run_path}/SVM/Predictions'):
    os.makedirs(f'{run_path}/SVM/Predictions')

# Load the labels
labels = data_file['label'].flatten()

# Split the training and validation data
training_data = embeds[train_mask]
val_data = embeds[val_mask]
training_labels = labels[train_mask]
val_labels = labels[val_mask]

# Concatenate the training and validation data 
# (We do it as later we use cross-validation to tune the hyperparameters and train the SVM)
training_data = np.concatenate([training_data, val_data], axis=0)
training_labels = np.concatenate([training_labels, val_labels], axis=0)


### DIMENSIONALITY REDUCTION ###
# Here we can apply different dimensionality reduction techniques to the embeddings before training the GMM
# The best results in our exepriments are obtained with the Autoencoder, 
# but it can also be use PCA, TSNE, LDA, Random Projection, SVD, etc.

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

# Initialize the MLP regressor that will be used as an Autoencoder
mlp = MLPRegressor(hidden_layer_sizes=(4, ), 
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

# Training the autoencoder with the normal class
# mlp.fit(training_data[training_labels == 0], training_data[training_labels == 0])

# Training the autoencoder with the whole training data
mlp.fit(training_data, training_data)

# Get the hidden layer of the autoencoder
mlp_hidden = mlp.coefs_[0]

# Apply the hidden layer to the training data to reduce the dimensionality
training_data = np.dot(training_data, mlp_hidden)

# Plot the loss curve
plt.plot(range(1, len(mlp.loss_curve_)+1), mlp.loss_curve_)
plt.savefig(f"{run_path}/SVM/Plots/MLP_Loss.png")
plt.close()


# Plot the features after dimensionality reduction
if training_data.shape[1] >= 2:
    # If it has more than 2 dimensions, we plot the first two
    plt.scatter(training_data[training_labels == 0][:, 0], training_data[training_labels == 0][:, 1], label='Normal')
    plt.scatter(training_data[training_labels == 1][:, 0], training_data[training_labels == 1][:, 1], label='Anomaly')
    plt.legend()
    plt.savefig(f"{run_path}/SVM/Plots/dimensionality_reduction.png")
    plt.close()


# Define the parameter grid for the SVM grid search
param_grid = {
    'C': [0.1, 1, 10, 100],# 1000],
    'gamma': [1, 0.1, 0.01, 0.001], # 0.0001],
    'kernel': ['rbf']
}

# Train the SVM with cross-validation to tune the hyperparameters
grid = GridSearchCV(SVC(), scoring='roc_auc', param_grid=param_grid, refit=True, verbose=3, n_jobs=16)
grid.fit(training_data, training_labels)

# Load the best model
best_model = grid.best_estimator_
print("Best parameters: ", grid.best_params_)


### Computing the test metrics
# Apply the dimensionality reduction to the test data
test_data = np.dot(embeds[test_mask], mlp_hidden)

# For the other dimensionality reduction techniques:
# test_data = tsne.transform(embeds[test_mask])
# test_data = pca.transform(embeds[test_mask])
# test_data = lda.transform(embeds[test_mask])
# test_data = rp.transform(embeds[test_mask])
# test_data = svd.transform(embeds[test_mask])

# Get the test labels, predictions and probabilities
test_labels = labels[test_mask]
test_preds = best_model.predict(test_data)
test_probs = best_model.decision_function(test_data)

# Computing test metrics, Recall, Precision and F1-score
print(classification_report(test_labels, test_preds))
report2save = classification_report(test_labels, test_preds, output_dict=True)

# ROC-AUC
roc_auc = roc_auc_score(test_labels, test_probs)
print("ROC-AUC: ", roc_auc)
report2save["ROC-AUC"] = roc_auc

# ROC curve test
fpr, tpr, thresholds = roc_curve(test_labels, test_probs)

# Plotitng the ROC curve
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
plt.savefig(f"{run_path}/SVM/Plots/ROC_Curve_TEST.png")
plt.close()


# Precision recall curve
precision, recall, thresholds = precision_recall_curve(test_labels, test_probs)

# Plotting the Precision-Recall curve
plt.figure()
lw = 2
plt.plot(
    recall,
    precision,
    color="darkorange",
    lw=lw,
    label="Precision-Recall curve",
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.savefig(f"{run_path}/SVM/Plots/Precision_Recall_Curve.png")
plt.close()

# Average Precision
average_precision = average_precision_score(test_labels, test_probs)
print("Average Precision: ", average_precision)
report2save["AveragePrecision"] = average_precision

# Save the test metrics
with open(f"{run_path}/SVM/Results/SVM_results.txt", "w") as f:
    f.write(str(report2save))

# Save test predictions as an .npy
test_preds = np.array(test_preds, dtype=int)
np.save(f"{run_path}/SVM/Predictions/test_preds.npy", test_preds)