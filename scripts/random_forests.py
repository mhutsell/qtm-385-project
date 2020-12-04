import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
DATA_PATH = "/Users/Mack/qtm-385-project/data/"
INT_PATH = "/Users/Mack/qtm-385-project/intermediate/"
umap_emb = pickle.load(open(INT_PATH + "normal_umap.p", "rb"))
pa_umap_emb = pickle.load(open(INT_PATH + "parametric_umap.p", "rb"))
data = pd.read_csv(DATA_PATH + "pca_data.csv")
ind = pickle.load(open(INT_PATH + "indices.p", "rb"))
rfc = RFC()
clusters = pd.read_csv(DATA_PATH + "phenograph_clusters.csv")

pops = clusters.iloc[ind]

for size in [5000,10000,15000,20000,25000,30000,35000,40000, 45000, 50000]:
	print("Testing Classification for UMAP vs pUMAP at {} samples".format(size))
	t_Y = pops[:size]
	te_Y = pops[100000-size:]
	train_Y = [int(x.split(";")[1]) for x in t_Y[';"x"']]
	test_Y = [int(x.split(";")[1]) for x in te_Y[';"x"']]
	for df_set in [(umap_emb, "Normal UMAP"), (pa_umap_emb, "Parametric UMAP")]:
		avg = 0
		tot_err = 0
		for x in range(3):
			df = df_set[0]
			method = df_set[1]
			train_X = df[:size]
			test_X = df[100000-size:]
			rfc.fit(train_X, train_Y)
			predict = rfc.predict(test_X)
			errors = 0
			for x in range(size):
				if predict[x] != test_Y[x]:
					errors += 1
			accuracy = 1.0 - (float(errors) / (size))
			avg += accuracy
		avg /= 3
		tot_err += errors
		print("{} Accuracy: {}, # Errors: {} / {}".format(method, avg, tot_err, size))
