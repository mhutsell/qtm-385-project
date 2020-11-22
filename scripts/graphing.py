import matplotlib.pyplot as plt
import pickle

data_path = "/Users/Mack/qtm-385-project/intermediate/"

param_umap = pickle.load(open(data_path + "parametric_umap.p", "rb"))
normal_umap = pickle.load(open(data_path + "normal_umap.p", "rb"))


plt.scatter(param_umap[:,0], param_umap[:,1])
plt.figure()
plt.scatter(normal_umap[:,0], normal_umap[:,1])
plt.show()
