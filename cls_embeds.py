import pickle as pkl
from scipy.io import loadmat

pickle_file_name = "embeds_contr_sup_drop=0.1_hidd=35_out=30_lr=0.001_model=GCN_Att"
# pickle_file_name = "embeddings_contrastive_dropout_augmentations"
with open(f'./Pickles/{pickle_file_name}.pkl', 'rb') as file:
    out = pkl.load(file)

with open('./Pickles/train_test_val_masks__contr_sup_drop=0.0_hidd=25_out=25_lr=0.001_model=Simpler_GCN.pkl', 'rb') as file:
    train_mask, val_mask, test_mask, train_mask_contrastive = pkl.load(file)


data_file = loadmat('./Data/Amazon.mat')
labels = data_file['label'].flatten()

train_labels = labels[train_mask]
val_labels = labels[val_mask]
test_labels = labels[test_mask]

X_train = out[train_mask]
X_val = out[val_mask]
X_test = out[test_mask]

# # Tsne
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(out)
# X_train = X_tsne[train_mask]
# X_val = X_tsne[val_mask]
# X_test = X_tsne[test_mask]


from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
print('Random Forest')

clf = RandomForestClassifier(n_estimators=150, class_weight='balanced')
clf.fit(X_train, train_labels)

y_pred = clf.predict(X_test)
print(classification_report(test_labels, y_pred))

print("SVM")
from sklearn.svm import SVC
clf = SVC(class_weight='balanced')
clf.fit(X_train, train_labels)

y_pred = clf.predict(X_test)
print(classification_report(test_labels, y_pred))

print("Gradient boosting")
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=150)
clf.fit(X_train, train_labels)

y_pred = clf.predict(X_test)

print(classification_report(test_labels, y_pred))

print("Original features")
feat_data = data_file['features'].todense().A

X_train = feat_data[train_mask]
X_val = feat_data[val_mask]
X_test = feat_data[test_mask]

print('Random Forest')
clf = RandomForestClassifier(n_estimators=150, class_weight='balanced')
clf.fit(X_train, train_labels)

y_pred = clf.predict(X_test)
print(classification_report(test_labels, y_pred))

print("SVM")
clf = SVC(class_weight='balanced')
clf.fit(X_train, train_labels)

y_pred = clf.predict(X_test)
print(classification_report(test_labels, y_pred))