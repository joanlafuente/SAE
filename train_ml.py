from scipy.io import loadmat
import pickle
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import *
import yaml
import sys

name_yaml = sys.argv[1]
print(f'Running {name_yaml}')

# Open a yaml file with the parameters
with open(f'./Setups/ML/{name_yaml}.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)
if params["data"] == "amz":
    run_path = f"./Runs/ML/Amazon/{name_yaml}"
if params["data"] == "Yelp":
    run_path = f"./Runs/ML/Yelp/{name_yaml}"

mat_path = params["mat_path"]
homo_adjlist = params["homo_adjlist"]

data_file = loadmat(mat_path)
labels = data_file['label'].flatten()
feat_data = data_file['features'].todense().A

with open(homo_adjlist, 'rb') as file:
    homo = pickle.load(file)
file.close()


# Assuming 'feat_data' and 'labels' are defined as per the context
# Combine features and labels into a single DataFrame
data = pd.DataFrame(feat_data[3305:, :], columns=[f'feature_{i}' for i in range(feat_data.shape[1])])
data['target'] = labels[3305:]

# Splitting the data into training and testing dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['target'])

# Setup PyCaret - target column is 'target'
clf1 = setup(data=train_data, target='target', session_id=42, use_gpu=True)

# Compare models
best_model = compare_models()

# Optionally, you can also evaluate the models on the test dataset
predict_model(best_model, data=test_data)


'''from sklearn.metrics import classification_report
report = classification_report(

with open(f'{run_path}/Report/cls_{name_yaml}.txt', 'w') as file:
    file.write(str(report))'''