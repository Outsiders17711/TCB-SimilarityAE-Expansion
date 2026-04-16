import scipy
import scipy.cluster.hierarchy as hc
from fastai.tabular.all import *  # type:ignore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .basics import *
from .modules import slicePalette, yamlLoader

__all__ = [
    "setReproducibility",
    "logTransform",
    "tabularDataloader",
    "featureClustering",
    "baseClusterAnalysis",
    "gridClusterAnalysis",
    "staticMultiClusterMap",
]


## -------------------------------------------------------------- ##
def setReproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    #
    g = torch.Generator()
    g.manual_seed(seed)

    #
    def e_seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    no_random(seed=17711, reproducible=True)
    return g, e_seed_worker


def logTransform(series: pd.Series, revert=False) -> pd.Series:
    """
    apply log-transform or its inverse (exponential) to the input series;
    `np.log1p` and `np.expm1` are used for symmetry and numerical stability when demand is zero.
    """
    series = series.copy()
    if revert:
        return np.expm1(series)  # type:ignore
    return np.log1p(series)  # type:ignore


## -------------------------------------------------------------- ##
def tabularDataloader(dset, *, f_outputs, f_inputs=None, normalised=False, pct_valid=0.0, verbose=False):
    f_ignores = yamlLoader("./schema/modelling.yaml")["ignores"]
    f_inputs = f_inputs or [c for c in dset.columns if c not in (f_outputs + f_ignores)]
    f_cats = dset[f_inputs].select_dtypes(exclude=np.number).columns.tolist()
    f_nums = dset[f_inputs].select_dtypes(include=np.number).columns.tolist()

    to = TabularPandas(
        dset[f_cats + f_nums + f_outputs],
        procs=[Categorify, Normalize] if normalised else [Categorify],
        cat_names=f_cats,
        cont_names=f_nums,
        y_names=f_outputs,
        splits=RandomSplitter(valid_pct=pct_valid)(range_of(dset)),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    print("to.train (X,y):", to.train.xs.shape, to.train.ys.shape)
    print("to.valid (X,y):", to.valid.xs.shape, to.valid.ys.shape)
    if verbose:
        print(f"{f_inputs=}", f"{f_cats=}", f"{f_nums=}", f"{f_outputs=}", sep="\n")
        print("-" * 69)
        _ = [print(i, end="") for i in to.procs]
        print("-" * 69)
        to.show(max_n=7)  # replaced `to.show_batch()` for working `max_n`
        print("-" * 69)

    return to


## -------------------------------------------------------------- ##
def featureClustering(df, figsize=(16, 10), font_size=11, save: str | None = None):
    """adapted from fastai's fastbook.cluster_columns"""
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method="average")

    plt.figure(figsize=figsize, dpi=300)
    hc.dendrogram(z, labels=df.columns, orientation="left", leaf_font_size=font_size)
    plt.xticks(fontsize=font_size - 1)
    plt.tight_layout()
    if save:
        # plt.title("Feature Clustering Dendrogram", fontsize=font_size + 3)
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        plt.close()
    plt.show()


class baseClusterAnalysis:
    """base class for streamlined clustering analysis of tcb grid features/encodings"""

    def __init__(self, features: pd.DataFrame):
        self.data = features.drop(columns="id")
        self.grids = features["id"]
        # ---
        self.random_state = 17711
        self.results = {}

    def _find_optimal_clusters(self, max_k):
        """find optimal number of clusters using silhouette score"""
        k_range = range(2, min(max_k + 1, len(self.data) // 10))
        silhouettes = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            silhouettes.append(score)

        best_k = k_range[np.argmax(silhouettes)]
        best_score = max(silhouettes)
        return best_k, best_score, list(k_range), silhouettes

    def _compute_kmeans(self, n_clusters):
        """compute k-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(self.data)

        self.results["kmeans"] = {
            "labels": labels,
            "n_clusters": n_clusters,
            "centers": kmeans.cluster_centers_,
            "silhouette": silhouette_score(self.data, labels),
        }

    def runAnalysis(self, max_k=5, best_k=None, dim_redux=False):
        """
        run complete clustering analysis, testing up to max_k clusters for best cluster size;
        can override the test by passing a desired cluster size to best_k.
        """
        self.dim_redux = dim_redux
        print("running feature clustering analysis...")

        if best_k is None:
            # find optimal clusters and compute k-means
            print("finding optimal clusters and computing k-means...")
            best_k, best_score, k_range, silhouettes = self._find_optimal_clusters(max_k)
        else:
            print("using provided cluster size for computing k-means...")
            k_range = silhouettes = []
            best_score = 0

        self._compute_kmeans(best_k)
        best_score = best_score or self.results["kmeans"]["silhouette"]

        self.results["optimisation"] = {
            "k_range": k_range,
            "silhouettes": silhouettes,
            "best_k": best_k,
            "best_score": best_score,
        }
        print(f"analysis complete. optimal clusters: {best_k} (silhouette: {best_score:.3f})")
        return self.results

    def plotResults(self, save: str | None = None):
        """plot clustering results"""
        assert self.results, "please run clustering analysis before plotting results"
        if not self.dim_redux:
            print("warning: t-sne/umap dimensionality reduction not performed; exiting...")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # silhouette scores
        opt = self.results["optimisation"]
        axes[0].plot(opt["k_range"], opt["silhouettes"], "bo-")
        axes[0].axvline(x=opt["best_k"], color="r", linestyle="--", alpha=0.7)
        axes[0].set_title(f"Optimal Clusters: {opt['best_k']} (Silhouette: {opt['best_score']:.3f})")
        axes[0].set_xlabel("Number of Clusters")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].grid(True, alpha=0.3)

        # cluster distribution
        kmeans = self.results["kmeans"]
        unique, counts = np.unique(kmeans["labels"], return_counts=True)
        axes[1].bar(unique, counts, alpha=0.7)
        axes[1].set_title(f"Cluster Distribution ({kmeans['n_clusters']} clusters)")
        axes[1].set_xlabel("Cluster ID")
        axes[1].set_ylabel("Number of Points")

        # t-sne and umap
        reductions = {"umap": ("UMAP", "n_neighbors"), "tsne": ("t-SNE", "perplexity")}
        for i, r_method in enumerate(reductions, start=2):
            r_name, r_param = reductions[r_method]
            coords = self.results[r_method]["coordinates"]
            p_value = self.results[r_method][r_param]

            axes[i].scatter(coords[:, 0], coords[:, 1], c=kmeans["labels"], cmap="Set3", alpha=0.7, s=1)
            axes[i].set_title(f"{r_name} ({r_param}={p_value})")
            axes[i].set_xlabel(f"{r_name} 1")
            axes[i].set_ylabel(f"{r_name} 2")

        plt.tight_layout()
        if save:
            fig.suptitle("TCB Grid Feature Clustering Analysis", fontsize=16)
            plt.tight_layout()  # re-adjust layout to prevent suptitle overlap with subplots
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save, dpi=300, bbox_inches="tight")
            print(f"clustering plots saved to: ./{save}")
            plt.close()
        plt.show()

    def saveResults(self):
        """return augmented y with cluster labels for visualisation"""
        assert self.results, "please run clustering analysis before plotting results"
        y_augmented = self.grids.copy().to_frame().astype(int)

        # add cluster labels to y
        y_augmented["cluster"] = self.results["kmeans"]["labels"]
        if self.dim_redux:
            y_augmented["tsne_x"] = self.results["tsne"]["coordinates"][:, 0]
            y_augmented["tsne_y"] = self.results["tsne"]["coordinates"][:, 1]
            y_augmented["umap_x"] = self.results["umap"]["coordinates"][:, 0]
            y_augmented["umap_y"] = self.results["umap"]["coordinates"][:, 1]

        return y_augmented


class gridClusterAnalysis(baseClusterAnalysis):
    """
    subclass of `baseClusterAnalysis` to analyse preprocessed features from a tabular object;
    scaling is not applied here as data is already normalised in tabularpandas.
    """

    def __init__(self, to: TabularPandas, features: list[str]):
        to = to.copy()
        self.data = to.items[features]  # get processed tabular data
        self.grids = to.new(to.all_cols).decode().items["id"]  # get original grid ids
        # ---
        self.random_state = 17711
        self.results = {}


def staticMultiClusterMap(
    grids: gpd.GeoDataFrame,
    clusters: pd.DataFrame,
    *,
    fn: str,
    title: str | None = None,
):
    """
    visualise multi-cluster analysis results on a static map;
    similar to `gridClusterAnalysisMap` but plots all available clusters.
    """
    gdf = grids.merge(clusters, on="id", how="left")
    border = gpd.GeoSeries(gdf.union_all(method="unary"), crs=gdf.crs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_axis_off()
    ax.margins(0.01)  # remove whitespace

    border.boundary.plot(ax=ax, color="black", linewidth=2)  # study area
    gdf.plot(ax=ax, color="lightgrey", markersize=1, alpha=0.25)  # full grid

    nc = clusters["cluster"].nunique()
    cmap = list(slicePalette("Set2", n=nc, reverse=False).colors)  # type:ignore
    for cid in range(nc):
        c = gdf[gdf["cluster"] == cid]
        c.plot(ax=ax, color=cmap[cid])
        plt.scatter([], [], color=cmap[cid], label=f"Cluster {cid + 1} ({len(c)})", s=9, marker="s")

    # plot existing bike stations for reference
    existing = gdf[gdf["isBikeStation"] == True].geometry.centroid
    existing.plot(ax=ax, color="blue", markersize=9, label=f"BSS Stations ({len(existing)})")

    plt.legend(fontsize=11, loc="lower right", bbox_to_anchor=(0.99, 0.0))
    plt.title(title, fontsize=9) if title else None
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
