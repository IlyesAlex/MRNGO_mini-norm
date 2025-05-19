import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from matplotlib.patches import Polygon
import hdbscan
from scipy.spatial import ConvexHull



concept_map = {
    "alma":        "apple",
    "asztal":      "table",
    "autó":        "car",
    "bab":         "bean",
    "ceruza":      "pencil",
    "cipő":        "shoe",
    "citrom":      "lemon",
    "eper":        "strawberry",
    "erdő":        "forest",
    "gomba":       "mushroom",
    "gyerek":      "child",
    "helikopter":  "helicopter",
    "hercegnő":    "princess",
    "játszótér":   "playground",
    "kacsa":       "duck",
    "kard":        "sword",
    "krumpli":     "potato",
    "kutya":       "dog",
    "kéz":         "hand",
    "könyv":       "book",
    "láb":         "foot",
    "ló":          "horse",
    "nadrág":      "trousers",
    "nappali":     "livingroom",
    "pulóver":     "sweater",
    "seprű":       "broom",
    "szem":        "eye",
    "szőnyeg":     "carpet",
    "tűzoltó":     "firefighter",
    "vonat":       "train",
}

category_colors = {
    "animals":    "#00c9c9",
    "bodyparts":  "#80ff99",
    "clothes":    "#c0e07e",
    "fooddrink":  "#9ecc27",
    "locations":  "#f4ce97",
    "people":     "#f2b64c",
    "plants":     "#f8c5c0",
    "tools":      "#f492a2",
    "toys":       "#f3705e",
    "vehicles":   "#e62d00",
}

#%% ── 0) LOAD YOUR VECTOR SPACE & META ─────────────────────────────────────────
vector_df_path = "../processed/MRNGO_mini-norm_vectorspace.xlsx"
select_sheet    = "mcrae_vector_matrix"  

# 0.1) read the frequency matrix, re-map row-index to English
df_vec = pd.read_excel(vector_df_path,
                       sheet_name=select_sheet,
                       index_col=0)
df_vec.index = df_vec.index.map(lambda c: concept_map.get(c, c))

# 0.2) read your category lookup (must already use English keys!)
meta = pd.read_csv("../processed/MRNGO_meta.csv", index_col=0, dtype=str)
# assume meta has columns: 'Concept' (English) and 'Category'
# re-index meta by the English concept names:
cats = meta["Category"]

# now subset cats to only those in df_vec
cats = cats.reindex(df_vec.index)

#%% ── 1) COMPUTE A 2D UMAP EMBEDDING ──────────────────────────────────────────
reducer = umap.UMAP(n_components=2, random_state=42)
emb = reducer.fit_transform(df_vec.values)

df_emb = pd.DataFrame(emb,
                      index=df_vec.index,
                      columns=["UMAP1","UMAP2"])

clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
# bring in the cluster & category & concept labels
df_emb['cluster']  = DBSCAN(eps=0.5, min_samples=2).fit_predict(df_emb[["UMAP1","UMAP2"]])
df_emb['cluster_hdb'] = clusterer.fit_predict(df_emb[["UMAP1","UMAP2"]])
df_emb['category'] = cats
df_emb['concept']  = df_emb.index

#%% ── 2) PLOT WITH TRANSLUCENT HULLS & MARKERS ───────────────────────────────
marker_list = ['o','s','^','v','P','X','D','*','h','+']
n_markers   = len(marker_list)

fig, ax = plt.subplots(figsize=(12,10))

# draw one convex‐hull per cluster (skip noise = -1)
for cl in sorted(df_emb['cluster'].unique()):
    if cl < 0:
        continue
    pts = df_emb.loc[df_emb['cluster']==cl, ['UMAP1','UMAP2']].values
    if pts.shape[0] >= 2:
        # 1) make a multipoint, take its convex hull, then buffer it
        mp = MultiPoint(pts)
        hull = mp.convex_hull.buffer(0.3, resolution=16)  
        #    ↑ tweak the 0.3 radius (and resolution) to taste
        
        # 2) extract x,y and draw
        xs, ys = hull.exterior.xy
        ax.fill(xs, ys,
                facecolor=f"C{cl%10}",
                edgecolor=None,
                alpha=0.2,
                zorder=1)

# scatter & annotate
for _, row in df_emb.iterrows():
    m = marker_list[row['cluster'] % n_markers] if row['cluster'] >= 0 else 'x'
    ax.scatter(row.UMAP1, row.UMAP2,
               marker=m,
               s=200,
               edgecolor='k',
               color=category_colors[row['category']],
               linewidth=0.8,
               zorder=3)
    ax.text(row.UMAP1,
            row.UMAP2 + 0.08,    # nudge label upward
            row['concept'],
            ha='center', va='bottom',
            fontsize=20,
            zorder=4)

ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
plt.tight_layout()
ax.axis('off')

plt.tight_layout(pad=0)
plt.show()

# save high-res for poster use
fig.savefig("../figures/umap_concept_space.png", dpi=300)
plt.show()

#%% OPTIMIZER

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

X = df_emb[['UMAP1','UMAP2']].values

best = {'score': -1, 'eps': None, 'min_samples': None}
for eps in np.linspace(0.1, 1.0, 10):
    for ms in [3, 5, 7, 10]:
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        # mask-out the noise
        non_noise = labels != -1
        unique_clusters = set(labels[non_noise])
        # only keep runs with >= 2 true clusters
        if len(unique_clusters) >= 2:
            # compute silhouette on non-noise points only
            score = silhouette_score(X[non_noise], labels[non_noise])
            if score > best['score']:
                best.update({'score': score, 'eps': eps, 'min_samples': ms})

print("Best valid Silhouette:", best)

