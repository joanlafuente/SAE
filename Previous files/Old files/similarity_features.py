import pickle as pkl
from scipy.io import loadmat

pickle_file_name = "embeds_contr_sup_drop=0.2_hidd=40_out=35_lr=0.001_model=GCN_Att_Yelp"
# pickle_file_name = "embeds_contr_sup_drop=0.1_hidd=35_out=30_lr=0.001_model=GCN_Att"
# pickle_file_name = "embeddings_contrastive_dropout_augmentations"
with open(f'./Pickles/{pickle_file_name}.pkl', 'rb') as file:
    out = pkl.load(file)

with open('./Pickles/train_test_val_masks__contr_sup_drop=0.2_hidd=35_out=30_lr=0.001_model=GCN_Att_Yelp.pkl', 'rb') as file:
    train_mask, val_mask, test_mask, train_mask_contrastive = pkl.load(file)


data_file = loadmat('./Data/YelpChi.mat')
labels = data_file['label'].flatten()
feat_data = data_file['features'].todense().A


train_labels = labels[train_mask]
val_labels = labels[val_mask]
test_labels = labels[test_mask]


X_test_new = out[test_mask]
X_test_old = feat_data[test_mask]


# Cosine similarity between the embeddings of the nodes within the same class
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_cosine_similarity(X, labels, class_label=1):
    X = X[labels == class_label]
    cos_sim = cosine_similarity(X)
    cos_sim = np.triu(cos_sim, k=1)
    cos_sim = cos_sim[cos_sim != 0]
    cos_sim = np.mean(cos_sim)
    return cos_sim

cos_sim_new_0 = get_cosine_similarity(X_test_new, test_labels, class_label=0)
cos_sim_old_0 = get_cosine_similarity(X_test_old, test_labels, class_label=0)

cos_sim_new_1 = get_cosine_similarity(X_test_new, test_labels, class_label=1)
cos_sim_old_1 = get_cosine_similarity(X_test_old, test_labels, class_label=1)

print("Cosine similarity between the embeddings of the nodes within the same class")
print("Class 0")
print("New:", cos_sim_new_0.mean())
print("Old:", cos_sim_old_0.mean())
print("Class 1")
print("New:", cos_sim_new_1.mean())
print("Old:", cos_sim_old_1.mean())

# Euclidean distance between the embeddings of the nodes within the same class
from sklearn.metrics.pairwise import euclidean_distances

def get_euclidean_distance(X, labels, class_label=1):
    X = X[labels == class_label]
    euclidean_dist = euclidean_distances(X)
    euclidean_dist = np.triu(euclidean_dist, k=1)
    euclidean_dist = euclidean_dist[euclidean_dist != 0]
    euclidean_dist = np.mean(euclidean_dist)
    return euclidean_dist

euclidean_dist_new_0 = get_euclidean_distance(X_test_new, test_labels, class_label=0)
euclidean_dist_old_0 = get_euclidean_distance(X_test_old, test_labels, class_label=0)

euclidean_dist_new_1 = get_euclidean_distance(X_test_new, test_labels, class_label=1)
euclidean_dist_old_1 = get_euclidean_distance(X_test_old, test_labels, class_label=1)

print("Euclidean distance between the embeddings of the nodes within the same class")
print("Class 0")
print("New:", euclidean_dist_new_0.mean())
print("Old:", euclidean_dist_old_0.mean())
print("Class 1")
print("New:", euclidean_dist_new_1.mean())
print("Old:", euclidean_dist_old_1.mean())

