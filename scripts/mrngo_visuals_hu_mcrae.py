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

concept_map_rev = {v: k for k, v in concept_map.items()}
concept_map_rev.update({
    "desk":        "asztal",
    "beans":       "bab",
    "shoes":       "cipő",
    "children":    "gyerek",
    "kid":         "gyerek",
    "leg":         "láb",
    "legs":        "láb",
    "pants":       "nadrág",
    "living room": "nappali",
    "living-room": "nappali",
    "jumper":      "pulóver",
    "pullover":    "pulóver",
    "eyes":        "szem",
    "rug":         "szőnyeg",
})

concept_map_en = {v: v for k, v in concept_map.items()}
concept_map_en.update({
    "desk":        "table",
    "beans":       "bean",
    "shoes":       "shoe",
    "children":    "child",
    "kid":         "child",
    "leg":         "foot",
    "legs":        "foot",
    "pants":       "trousers",
    "living room": "livingroom",
    "living-room": "livingroom",
    "jumper":      "sweater",
    "pullover":    "sweater",
    "eyes":        "eye",
    "rug":         "carpet",
})

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


def concept_to_english(concept):
    concept = str(concept).lower()
    return concept_map.get(concept, concept_map_en.get(concept, concept))


def concept_to_hungarian(concept):
    concept = str(concept).lower()
    return concept_map_rev.get(concept, concept)


def plot_umap_concept_space(df_vec, output_path, label_language="en", concepts_to_plot=None):
    # -- 0) LOAD META & SET CONCEPT LABELS --------------------------------------
    concept_labels_hu = df_vec.index.map(concept_to_hungarian)
    concept_labels_en = df_vec.index.map(concept_to_english)
    concept_labels = concept_labels_hu if label_language == "hu" else concept_labels_en

    # 0.1) read your category lookup (must already use English keys!)
    meta = pd.read_csv("../processed/MRNGO_meta.csv", index_col=0, dtype=str)
    # assume meta has columns: 'Concept' (English) and 'Category'
    # re-index meta by the English concept names:
    cats = meta["Category"]

    # now subset cats to only those in df_vec
    cats = cats.reindex(concept_labels_en)

    # -- 1) COMPUTE A 2D UMAP EMBEDDING -----------------------------------------
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(df_vec.values)

    df_emb = pd.DataFrame(emb,
                          index=df_vec.index,
                          columns=["UMAP1","UMAP2"])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    # bring in the cluster & category & concept labels
    df_emb['cluster']  = DBSCAN(eps=0.5, min_samples=2).fit_predict(df_emb[["UMAP1","UMAP2"]])
    df_emb['cluster_hdb'] = clusterer.fit_predict(df_emb[["UMAP1","UMAP2"]])
    df_emb['category'] = cats.values
    df_emb['concept']  = concept_labels.values

    if concepts_to_plot is None:
        df_plot = df_emb
    else:
        concepts_to_plot = set(concepts_to_plot)
        df_plot = df_emb[df_emb.index.map(concept_to_english).isin(concepts_to_plot)]

    # -- 2) PLOT WITH TRANSLUCENT HULLS & MARKERS -------------------------------
    marker_list = ['o','s','^','v','P','X','D','*','h','+']
    n_markers   = len(marker_list)

    fig, ax = plt.subplots(figsize=(12,10))

    # draw one convex-hull per cluster (skip noise = -1)
    for cl in sorted(df_plot['cluster'].unique()):
        if cl < 0:
            continue
        pts = df_plot.loc[df_plot['cluster']==cl, ['UMAP1','UMAP2']].values
        if pts.shape[0] >= 2:
            # 1) make a multipoint, take its convex hull, then buffer it
            mp = MultiPoint(pts)
            hull = mp.convex_hull.buffer(0.3, resolution=16)
            # tweak the 0.3 radius (and resolution) to taste

            # 2) extract x,y and draw
            xs, ys = hull.exterior.xy
            ax.fill(xs, ys,
                    facecolor=f"C{cl%10}",
                    edgecolor=None,
                    alpha=0.2,
                    zorder=1)

    # scatter & annotate
    for _, row in df_plot.iterrows():
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
    fig.savefig(output_path, dpi=300)
    plt.show()


#%% MCRAE VECTOR SPACE
mcrae_norm_longformat = pd.read_excel("../others/McRae-norm_filtered.xlsx")
mcrae_norm_filtered = mcrae_norm_longformat[
    mcrae_norm_longformat["Concept"].map(lambda x: str(x).lower()).isin(concept_map_rev.keys())
]
mcrae_norm_filtered = mcrae_norm_filtered[
    mcrae_norm_filtered["Concept"].map(lambda x: str(x).lower()) != "desk"
]

mcrae_vector_matrix = mcrae_norm_filtered.pivot_table(
    index="Concept",
    columns="Feature",
    values="Prod_Freq",
    aggfunc="sum",
    fill_value=0
)

with pd.ExcelWriter("../others/McRae-vectorspace.xlsx", engine="openpyxl") as writer:
    mcrae_vector_matrix.to_excel(writer, sheet_name="mcrae_vector_matrix", index=True)

#%% MRNGO WITH ENGLISH LABELS
vector_df_path = "../processed/MRNGO_mini-norm_vectorspace_new.xlsx"
select_sheet    = "mcrae_vector_matrix"

df_vec = pd.read_excel(vector_df_path,
                       sheet_name=select_sheet,
                       index_col=0)

mrngo_concepts_en = df_vec.index.map(concept_to_english)
mcrae_concepts_en = mcrae_vector_matrix.index.map(concept_to_english)
#df_vec = df_vec[mrngo_concepts_en.isin(mcrae_concepts_en)]
shared_concepts = mrngo_concepts_en[mrngo_concepts_en.isin(mcrae_concepts_en)]

plot_umap_concept_space(
    df_vec=df_vec,
    output_path="../figures/umap_concept_space_en.png",
    label_language="en",
    concepts_to_plot=shared_concepts
)


#%% MCRAE WITH ENGLISH LABELS
df_vec_mcrae = pd.read_excel("../others/McRae-vectorspace.xlsx",
                             sheet_name="mcrae_vector_matrix",
                             index_col=0)

plot_umap_concept_space(
    df_vec=df_vec_mcrae,
    output_path="../figures/umap_concept_space_mcrae_en.png",
    label_language="en"
)
