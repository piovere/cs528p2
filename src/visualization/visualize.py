from src.features.build_features import PCA
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
with sns.axes_style("darkgrid"):
    ax = sns.relplot(x=0, y=1, data=df_pca)
    ax.set_xlabels("Principal component 0")
    ax.set_ylabels("Principal Component 1")
    label_point(df_pca[0], df_pca[1], df_pca.School, plt.gca())
plt.savefig(os.path.join(plots_dir, "pc1_v_pc2.png"), dpi=300)
plt.clf()
