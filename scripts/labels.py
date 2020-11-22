from sklearn.cluster import KMeans
import pandas as pd
import pickle

DATA_PATH = "/Users/Mack/qtm-385-project/data/pca_data.csv"
UMAP_PATH = "/Users/Mack/qtm-385-project/intermediate/normal_umap.p"
PARAM_UMAP_PATH = "/Users/Mack/qtm-385-project/intermediate/parametric_umap.p"
INTERMEDIATE_PATH = "/Users/Mack/qtm-385-project/intermediate/"

df = pd.read_csv(DATA_PATH)
df = df.drop(["Unnamed: 0"], axis=1)

umap_emb = pickle.load(open(UMAP_PATH, "rb"))
param_umap_emb = pickle.load(open(PARAM_UMAP_PATH, "rb"))

kmeans_u = KMeans(n_clusters=100)
kmeans_u.fit(umap_emb)
umap_km_labels = kmeans_u.predict(umap_emb)
pickle.dump(umap_km_labels, open(INTERMEDIATE_PATH + "umap_km_labels.p", "wb"))

kmeans_p = KMeans(n_clusters=100)
kmeans_p.fit(param_umap_emb)
param_umap_km_labels = kmeans_p.predict(param_umap_emb)
pickle.dump(param_umap_km_labels, open(INTERMEDIATE_PATH + "param_umap_km_labels.p", "wb"))

kmeans_t = KMeans(n_clusters=100)
kmeans_t.fit(df)
fullData = kmeans_t.predict(df)
pickle.dump(fullData, open(INTERMEDIATE_PATH + "full_data_km_labels.p", "wb"))

print(param_umap_km_labels)
print(umap_km_labels)
print(fullData)
