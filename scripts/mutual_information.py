import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

INTERMEDIATE_PATH = "/Users/Mack/qtm-385-project/intermediate/"

umap_km_labels = pickle.load(open(INTERMEDIATE_PATH + "umap_km_labels.p", "rb"))
param_umap_km_labels = pickle.load(open(INTERMEDIATE_PATH + "param_umap_km_labels.p", "rb"))

fullData = pickle.load(open(INTERMEDIATE_PATH + "full_data_km_labels.p", "rb"))

inds = pickle.load(open(INTERMEDIATE_PATH + "indices.p", "rb"))

fullData = fullData[inds]

n_bins = 100
px_hist = plt.hist(fullData,n_bins,density=True)[0]
plt.title("Full Data K-Means")
plt.figure()

py_hist = plt.hist(umap_km_labels,n_bins,density=True)[0]
plt.title("UMAP Emb. Data K-Means")
plt.figure()

pxy_hist = plt.hist2d(fullData,umap_km_labels,bins=(n_bins,n_bins),density=True)[0]
plt.title("UMAP Emb. Data vs Full Data K-Means")
plt.figure()

py2_hist = plt.hist(param_umap_km_labels,n_bins,density=True)[0]
plt.title("pUMAP Emb. Data K-Means")
plt.figure()

pxy2_hist = plt.hist2d(fullData,param_umap_km_labels,bins=(n_bins,n_bins),density=True)[0]
plt.title("pUMAP Emb. Data vs Full Data K-Means")
plt.show()

print("Beginning UMAP Normal")
umap_sum = normalized_mutual_info_score(umap_km_labels, fullData)
pickle.dump(umap_sum, open(INTERMEDIATE_PATH + "umap_mutual.p", "wb"))
print(umap_sum)

print("Beginning UMAP Parametric")
param_umap_sum = normalized_mutual_info_score(param_umap_km_labels, fullData)
pickle.dump(param_umap_sum, open(INTERMEDIATE_PATH + "param_umap_mutual.p", "wb"))
print(param_umap_sum)

print("Beginning UMAP Normal")
umap_sum = adjusted_mutual_info_score(umap_km_labels, fullData)
pickle.dump(umap_sum, open(INTERMEDIATE_PATH + "umap_mutual.p", "wb"))
print(umap_sum)

print("Beginning UMAP Parametric")
param_umap_sum = adjusted_mutual_info_score(param_umap_km_labels, fullData)
pickle.dump(param_umap_sum, open(INTERMEDIATE_PATH + "param_umap_mutual.p", "wb"))
print(param_umap_sum)
