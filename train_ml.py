from scipy.io import loadmat
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, predict_model
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
train_data, test_data = train_test_split(data, train_size=0.7, random_state=0, stratify=data['target'])

# Setup PyCaret - target column is 'target'
clf1 = setup(data=train_data, target='target', session_id=42, use_gpu=True)

# Compare models
best_model = compare_models()

# Evaluate the best model on the test dataset
predictions = predict_model(best_model, data=test_data)

# Print the columns of the predictions DataFrame
print(predictions.columns)

# Extract true labels and predicted labels based on the actual column names
y_true = test_data['target']
y_pred = predictions['Label'] if 'Label' in predictions.columns else predictions['prediction_label']

# Generate classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Optionally, save the report to a CSV file
report_df.to_csv(f"{run_path}_classification_report.csv", index=True)
