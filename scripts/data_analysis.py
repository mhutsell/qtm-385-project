import parametric_umap
import umap
import pandas as pd
from numpy import random
import numpy as np
import time
import pickle

data_path = "/Users/Mack/qtm-385-project/data/pca_data.csv"
SAVE_PATH = "/Users/Mack/qtm-385-project/intermediate"

data = pd.read_csv(data_path)
df = data.drop(["Unnamed: 0"], axis=1)

N = 100000
idx = random.choice(range(np.shape(df)[0]),N)
df = df.iloc[idx]

print("Saving chosen indices")
pickle.dump(idx, open(SAVE_PATH + "/indices.p", "wb"))

nn = 30
md = .2
seed = 123

pm = parametric_umap.ParametricUMAP(n_components=2, n_neighbors=nn, random_state=seed,min_dist=md, metric = "euclidean")

print("Beginning Parametric UMAP")
initial = time.time()
Y = pm.fit_transform(df)
finish = time.time()
total_time = finish - initial
print("Finished Parametric UMAP: {}".format(total_time))

pickle.dump(Y, open(SAVE_PATH + "/parametric_umap.p", "wb"))
pickle.dump(total_time, open(SAVE_PATH + "/parametric_umap_time.p", "wb"))

um = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=md, random_state=seed,metric="euclidean")

print("Beginning Normal UMAP")
initial = time.time()
Y2 = um.fit_transform(df)
finish = time.time()
total_time = finish - initial
print("Finished Normal UMAP: {}".format(total_time))

pickle.dump(Y2, open(SAVE_PATH + "/normal_umap.p", "wb"))
pickle.dump(total_time, open(SAVE_PATH + "/normal_umap_time.p", "wb"))

