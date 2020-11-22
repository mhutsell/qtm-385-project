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
t_Y = pops[:25000]
te_Y = pops[75000:]
train_Y = [int(x.split(";")[1]) for x in t_Y[';"x"']]
test_Y = [int(x.split(";")[1]) for x in te_Y[';"x"']]

for df_set in [(umap_emb, "Normal UMAP"), (pa_umap_emb, "Parametric UMAP")]:
	df = df_set[0]
	method = df_set[1]
	train_X = df[:25000]
	test_X = df[75000:]
	rfc.fit(train_X, train_Y)
	predict = rfc.predict(test_X)
	errors = 0
	for x in range(25000):
		if predict[x] != test_Y[x]:
			errors += 1
	accuracy = 1.0 - (float(errors) / 25000)
	print("{} Accuracy: {}, # Errors: {} / {}".format(method, accuracy, errors, len(train_X)))
