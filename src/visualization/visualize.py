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
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.95, linestyle=":", color='r', alpha=0.5)
plt.xlabel(r"Principal Components included in model, $k$")
plt.ylabel("Cumulative Variance Explained")
plt.savefig(os.path.join(plots_dir, 
                         "cumulative_explained_variance.png"), dpi=300)
plt.clf()
