from src.features.build_features import PCA, KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import os.path


# Images directory
plots_dir = os.path.join("reports", "figures")
# Load the data
datafile = "data/processed/UTK-peers_processed.csv"
df = pd.read_csv(datafile)
df_num = df.select_dtypes(exclude=['object'])

pca = PCA()
pca.fit(df_num.values)

# Plot a scree graph of singular values
plt.plot(pca.explained_variance_)
plt.xlabel("Principal component")
plt.ylabel(r"Explained variance ($s^2$)")
plt.savefig(os.path.join(plots_dir, "explained_variance.png"), dpi=300)
plt.clf()

# Plot of cumulative explained variance
f, ax = plt.subplots()
ax.plot(range(1, pca.explained_variance_.shape[0]+1),
        np.cumsum(pca.explained_variance_ratio_), '*-')
ax.axhline(y=0.95, linestyle=":", color='r', alpha=0.5)
ax.axvline(x=pca.to_explain_variance_frac(0.95)+1, linestyle=":", color='r',
            alpha=0.5)
ax.set_xlabel(r"Principal Components included in model, $k$")
ax.set_ylabel("Cumulative Variance Explained")
ax.annotate(f"{pca.to_explain_variance_frac(0.95) + 1} "
            f"components required \nfor $\geq$95% variance", 
            xy=(pca.to_explain_variance_frac(0.95)+1, 0.95), xycoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            xytext=(0.75, 0.25), textcoords='axes fraction',
            horizontalalignment="center")
plt.savefig(os.path.join(plots_dir, 
                         "cumulative_explained_variance.png"), dpi=300)
plt.clf()

# Scatter plot of the PC's against each other
pca2 = PCA(variance=0.95).fit(df_num.values)
df_pca = pd.DataFrame(pca2.transform(df_num.values))
df_pca['School'] = df["Name"]

# Code to label points obtained from 
# https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']), 
                fontdict={'size': 4})

f = plt.figure(figsize=(20,20))
with sns.axes_style("whitegrid"):
    ax = sns.relplot(x=0, y=1, data=df_pca)
    ax.set_xlabels("Principal component 0")
    ax.set_ylabels("Principal Component 1")
    label_point(df_pca[0], df_pca[1], df_pca.School, plt.gca())
plt.savefig(os.path.join(plots_dir, "pc1_v_pc2.png"), dpi=300)
plt.clf()

# Clustering of raw data
# Find optimal number of clusters
k_array = [_ for _ in range(2, df_num.shape[0])]
dunn_score = []
for k in k_array:
    kmt = KMeans()
    kmt.fit(df_num.values, k)
    di = kmt.dunn_index(df_num.values)
    dunn_score.append(di)

f = plt.figure()
plt.plot(k_array, dunn_score)
plt.xlabel("Number of clusters")
plt.ylabel("Dunn Index")
best_di = max(dunn_score)
best_k = k_array[dunn_score.index(best_di)]
plt.plot(best_k, best_di, 'ro')
plt.annotate(
    f"{best_k} clusters",
    xy=(best_k, best_di), xycoords='data',
    arrowprops=dict(facecolor='black', shrink=0.05),
    xytext=(0.75, 0.75), textcoords='axes fraction',
    horizontalalignment="center"
)
plt.savefig(os.path.join(plots_dir, "dunn_index_raw_data.png"), dpi=300)
plt.clf()

km = KMeans()
km.fit(df_num.values, k=13)
labels = km.predict(df_num)
f = plt.figure(figsize=(20,20))
with sns.axes_style("whitegrid"):
    ax = sns.relplot(x=0, y=1, data=df_pca, hue=labels)
    ax.set_xlabels("Principal component 0")
    ax.set_ylabels("Principal Component 1")
    label_point(df_pca[0], df_pca[1], df_pca.School, plt.gca())
plt.savefig(os.path.join(plots_dir, "pc1_v_pc2_raw_hue.png"), dpi=300)
plt.clf()
