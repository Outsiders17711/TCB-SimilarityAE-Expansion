import heapq
from contextlib import contextmanager

from scipy.spatial.distance import cdist
from shapely import wkt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from .basics import *

__all__ = [
    "aeSimilarityExpansion",
    "aeStationAllocation",
    "pipeCellOutput",
    "normaliseEncodings",
    "setDisplayOptions",
    "gdf2csv",
    "csv2gdf",
]


@contextmanager
def pipeCellOutput(file: str | None = None, mode="w"):
    """redirect output to a file, or discard if file is None"""
    pipe = file if file else os.devnull
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)

    bak_stdout = sys.stdout
    with open(pipe, mode) as f:
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = bak_stdout

    print(f"cell output piped to ./{file}") if file else None


def setDisplayOptions(font=False):
    np.set_printoptions(precision=2, linewidth=105)
    pd.options.display.max_rows = 7
    pd.options.display.max_columns = 13
    pd.options.display.precision = 2
    pd.options.display.float_format = "{:.2f}".format
    if font:
        plt.rcParams.update({"font.family": "Charis SIL"})


def normaliseEncodings(data, features: list[str], zscore: bool, l2norm: bool):
    """normalises encodings with optional z-scoring and/or l2 normalisation"""
    data = data.copy()
    Z = data[features].values

    if zscore:
        Z = (Z - Z.mean()) / Z.std(ddof=0)

    if l2norm:
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / np.maximum(norms, 1e-9)  # avoid division by zero

    data[features] = Z
    return data


def gdf2csv(gdf, fp):
    df = pd.DataFrame(gdf.copy())
    df["geometry"] = df["geometry"].apply(lambda geom: geom.wkt)  # type:ignore
    df.to_csv(fp, index=False, float_format="%.16g")


def csv2gdf(fp, crs="EPSG:25833"):
    df = pd.read_csv(fp)
    df["geometry"] = df["geometry"].apply(wkt.loads)  # type:ignore
    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)


class aeSimilarity:
    """computes per-candidate weights from latent encodings, measuring similarity to existing stations"""

    def __init__(
        self,
        Z: np.ndarray,
        existing: np.ndarray,
        candidates: np.ndarray,
        metric: Literal["cosine", "euclidean"] | str,
    ):
        self.Z = Z
        self.metric = metric

        self.n_existing = len(existing)
        self.n_candidates = len(candidates)
        self.E = self.Z[existing]
        self.C = self.Z[candidates]
        self.S = self.similarity(self.C, self.E)

    def similarity(self, C: np.ndarray, E: np.ndarray) -> np.ndarray:
        """return similarity matrix between candidate and existing stations"""
        if self.metric == "cosine":
            return C @ E.T
        elif self.metric == "euclidean":
            c2 = np.sum(C * C, axis=1, keepdims=True)
            e2 = np.sum(E * E, axis=1, keepdims=True).T
            return -(c2 + e2 - 2.0 * (C @ E.T))
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

    def nearest(self) -> np.ndarray:
        """weight is maximum similarity to any existing station"""
        return self.S.max(axis=1)

    def topk(self, k: int = 3) -> np.ndarray:
        """weight is mean of top-k similarities to existing stations"""
        if k == 1:
            print("aeSimilarity.topk(): k=1, using nearest neighbour similarity")
            return self.nearest()

        k = int(max(1, min(k, self.S.shape[1])))
        idx = np.argsort(self.S, axis=1)[:, -k:]
        return np.take_along_axis(self.S, idx, axis=1).mean(axis=1)

    def kde(self, sigma: float | None = None) -> np.ndarray:
        """weight is rbf kernel density estimate with respect to existing stations"""
        if sigma is None:
            rs = np.random.RandomState(17711)
            m = min(self.n_existing, 256)
            n = min(self.n_candidates, 256)
            Es = self.E[rs.choice(len(self.E), size=m, replace=False)]
            Cs = self.C[rs.choice(len(self.C), size=n, replace=False)]

            if self.metric == "euclidean":
                D2 = -self.similarity(Cs, Es)  # negate to get positive squared distances
            else:
                D2 = 2.0 - (2.0 * self.similarity(Cs, Es))

            sigma = np.sqrt(max(np.median(D2), 1e-9))
            print(f"aeSimilarity.kde(): estimated {sigma=:.4f} using median heuristic")

        gamma = 1.0 / (2.0 * sigma * sigma)  # type:ignore
        D2 = -self.S if self.metric == "euclidean" else 2.0 - 2.0 * self.S
        return np.exp(-gamma * D2).sum(axis=1)


class aeStationAllocation:
    """
    station allocation algorithm based on trained encodings from a hybrid denoising autoencoder;
    selects `n` new station locations using a similarity-based greedy algorithm with local search;
    option for excluding candidates within a buffer radius of existing stations;
    assumes the encodings have been preprocessed as desired (z-normalized, l2-normalized, etc.)
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        f_encodings: list,
        n: int,
        *,
        r_buffer: int = 250,
        method: Literal["topk", "kde"] = "topk",
        metric: Literal["cosine", "euclidean"] = "cosine",
        topk: int | None = 3,
        sigma: float | None = None,
    ) -> None:
        assert str(gdf.crs) == "EPSG:25833", "geometry must be in projected crs EPSG:25833 (metres)"
        assert method in {"topk", "kde"}, "method must be one of {'topk','kde'}"
        assert metric in {"cosine", "euclidean"}, "metric must be one of {'cosine','euclidean'}"
        assert n > 0, "n must be a positive integer"
        assert "isBikeStation" in gdf.columns, "gdf must contain 'isBikeStation' column"

        self.gdf = gdf
        self.Z = gdf[f_encodings].values

        self.n, self.r_buffer = n, r_buffer
        self.method, self.metric = method, metric
        self.topk, self.sigma = topk, sigma
        self.nopriors = False  # placeholder for compatibility; ae allocation requires priors

    def runAllocation(self) -> List[int]:
        """returns list of selected candidate grid ids"""
        self.buildConflictGraph()

        S = aeSimilarity(self.Z, self.bcgExisting, self.bcgCandidates, self.metric)
        self.topk = self.topk or len(self.bcgExisting)
        self.sWeights = S.topk(k=self.topk) if self.method == "topk" else S.kde(sigma=self.sigma)

        # initial greedy selection; then local search for improvement
        selected = self.greedySelection()  # indices in candidate space
        if len(selected) > 0:
            selected = self.swapLocalSearch(selected)

        self.idxSelected = selected = self.bcgCandidates[selected]  # indices in gdf space
        assert self.validateSelection(), "!!!"
        return self.gdf.loc[selected, "id"].tolist()

    def buildConflictGraph(self) -> None:
        """
        filters candidates within a buffer radius of existing stations;
        builds a conflict graph among remaining candidates grids (edges within buffer radius).

        returns:
        - indices of existing stations and candidate grids (in gdf index space)
        - indices of neighboring grids for each candidate (in candidate index space)
        """
        leaf = 20  # hard-code leaf size for efficiency with 35k+ grids
        buffer = self.r_buffer
        centroids = self.gdf.geometry.centroid
        centroids = np.vstack([centroids.x, centroids.y]).T

        # extract existing stations and candidates (in gdf index space)
        self.idxExisting = existing = self.gdf.index[self.gdf["isBikeStation"] == True].values
        if self.nopriors:
            candidates = self.gdf.index.values
            print(f"buildConflictGraph: nopriors mode - ignoring {len(existing)} existing stations")
            print(f"buildConflictGraph: working with {len(candidates)=} candidates (all grids)")

        else:
            candidates = self.gdf.index[self.gdf["isBikeStation"] == False].values
            if len(existing) == 0:
                raise ValueError("no existing stations found in gdf")
            print(f"buildConflictGraph: working with {len(existing)=} and {len(candidates)=} stations")

            # filter candidates too close to existing stations
            bt_existing = BallTree(centroids[existing], leaf_size=leaf, metric="euclidean")
            eqr = bt_existing.query_radius(centroids[candidates], r=buffer, return_distance=False)
            mask = np.array([len(ix) == 0 for ix in eqr])
            candidates = candidates[mask]  # candidates far enough from existing stations
            print(f"buildConflictGraph: {len(candidates)=} candidates remain after filtering")

        # build conflict graph among remaining candidates
        if len(candidates) == 0:
            raise ValueError(f"no candidates remain after filtering with buffer radius {buffer}m")
        else:
            bt_candidates = BallTree(centroids[candidates], leaf_size=leaf, metric="euclidean")
            # each list contains indices (in candidate index space) of neighbors for that candidate
            cqr = bt_candidates.query_radius(centroids[candidates], r=buffer, return_distance=False)
            neighbors = [set(int(j) for j in arr if int(j) != i) for i, arr in enumerate(cqr)]

        # indices of existing stations (gdf space)
        self.bcgExisting: np.ndarray = existing
        # indices of candidate locations after proximity filtering (gdf space)
        self.bcgCandidates: np.ndarray = candidates
        # indices of neighboring candidates for each candidate (candidate space)
        self.bcgNeighbors: List[Set[int]] = neighbors

    def greedySelection(self) -> List[int]:
        """
        greedy maximum-weight independent set on a geometric conflict graph.
        - weights: per-candidate weights, aligned with neighbours list (length M)
        - neighbours: adjacency as list of sets for indices 0..M-1
        returns a list of selected candidate indices (in candidate index space).
        """
        weights, neighbors = self.sWeights, self.bcgNeighbors
        M = len(weights)

        heap = [(-float(weights[i]), int(i)) for i in range(M)]
        heapq.heapify(heap)

        status = np.zeros(M, dtype=np.int8)
        selected = []

        while heap and len(selected) < self.n:
            _, i = heapq.heappop(heap)
            if status[i] == 0:  # available
                status[i] = 2  # selected
                selected.append(i)
                for j in neighbors[i]:
                    if status[j] == 0:
                        status[j] = 1  # banned

        assert selected, "greedySelection: no candidates selected, check inputs!"
        if len(selected) < self.n:
            print(f"greedySelection: only selected {len(selected)}/{self.n} requested candidates")
        return selected

    def swapLocalSearch(self, selected: List[int], max_iters: int = 10) -> List[int]:
        """
        1-for-1 swap hill-climbing local search to improve greedy selection
        returns possibly improved list of selected candidate indices (in candidate index space).
        """
        pool_size = 15000  # limit pool size for quality optimisation with 35k+ grids
        weights, neighbors = self.sWeights, self.bcgNeighbors

        # build pool of non-selected candidates sorted by weight
        ss_selected = set(selected)
        nonselected = np.array([i for i in range(len(weights)) if i not in ss_selected])
        if len(nonselected) == 0:  # all candidates selected; nothing to swap with
            return selected[:]

        # limit pool to top candidates by weight (based on pool_size)
        if len(nonselected) > pool_size:
            top = np.argsort(weights[nonselected])[-pool_size:]
            pool = set(nonselected[top])
        else:
            pool = set(nonselected.tolist())

        def _union_of_neighbours(indices: Set[int]) -> Set[int]:
            """get union of all neighbors of given indices (in candidate space)"""
            union = set()
            for i in indices:
                union.update(neighbors[i])
            return union

        for iter in range(max_iters):
            improved = False
            ll_selected = sorted(ss_selected)

            # try each selected candidate for replacement
            for sc in ll_selected:
                # candidates forbidden if they conflict with any other selected candidate
                osc = ss_selected - {sc}
                forbidden = _union_of_neighbours(osc) | osc

                # find feasible candidates from pool
                feasibles = [j for j in pool if j not in forbidden]
                if not feasibles:
                    continue

                # choose best feasible candidate by weight
                bfc = max(feasibles, key=lambda j: weights[j])

                # accept swap if it improves total weight
                wbfc, wsc = weights[bfc], weights[sc]
                if wbfc > wsc + 1e-12:
                    ss_selected.remove(sc)
                    ss_selected.add(bfc)

                    # update pool: put removed sc back and take added bfc out
                    pool.add(sc)
                    pool.discard(bfc)
                    improved = True

                    print(f"swapLocalSearch: swapped {sc} ({wsc:.4f}) with {bfc} ({wbfc:.4f})")
                    break  # break from inner loop and restart with the new selection

            if not improved:
                break

        return sorted(ss_selected, key=lambda i: -weights[i])

    def validateSelection(self) -> bool:
        """validates selected stations are not within buffer radius of existing stations or each other"""
        reference = self.gdf.loc[self.idxExisting].copy()
        selected = self.gdf.loc[self.idxSelected].copy()
        g_reference = np.array([p.coords[0] for p in reference.geometry.centroid])
        g_selected = np.array([p.coords[0] for p in selected.geometry.centroid])
        isValid = True

        # check distance to existing stations
        if not self.nopriors:
            t_reference = BallTree(g_reference, metric="euclidean")
            q_distances, q_indices = t_reference.query(g_selected, k=1)
            for i, dist in enumerate(q_distances):
                if dist[0] < self.r_buffer:
                    print(f"station {i} is too close to existing stations ({dist[0]:.2f})")
                    isValid = False

        # check distance between selected stations
        t_selected = BallTree(g_selected, metric="euclidean")
        # query for 2 nearest neighbors; the first is always the point itself
        q_distances, q_indices = t_selected.query(g_selected, k=2)

        skip = []  # keep track of already reported pairs
        # the nearest neighbor is in the second column (k=2)
        for i, (dist, idx) in enumerate(zip(q_distances[:, 1], q_indices[:, 1])):
            if (dist < self.r_buffer) and ([idx, i] not in skip):
                skip.append([i, idx])
                print(f"stations {i} and {idx} are too close to each other ({dist:.2f})")
                isValid = False

        print(f"validateSelection: validation check completed, {isValid=}")
        return isValid


class extendExistingStations:
    """
    consensus-based station allocation using DBSCAN clustering on candidates from multiple methods;
    extracts top candidates from each method, filters by proximity to existing stations,
    clusters remaining candidates, ranks clusters by method diversity, and selects top-n extensions.
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        n: int,
        *,
        s_methods: list[str],
        r_buffer: int = 250,
        d_epsilon: int = 500,
        min_candidates: int = 2,
        use_medoid: bool = True,
    ) -> None:
        assert str(gdf.crs) == "EPSG:25833", "geometry must be in projected crs EPSG:25833 (metres)"
        assert n > 0, "n must be a positive integer"

        self.gdf = gdf
        self.n, self.r_buffer = n, r_buffer
        self.methods = [m.upper() for m in s_methods]
        self.d_epsilon = d_epsilon
        self.min_candidates = min_candidates
        self.use_medoid = use_medoid

    def setup(self) -> None:
        self.gridCentroids = np.zeros(0)  # placeholder to avoid attribute errors in base class
        msg = "setup() must be implemented in subclass to run checks and precompute grid centroids"
        raise NotImplementedError(msg)

    def buildConflictGraph(self) -> None:
        leaf = 20  # hard-code leaf size for efficiency with 35k+ grids

        # extract existing stations and candidates (in gdf index space)
        existing = self.gdf.index[self.gdf["isBikeStation"] == True].values
        pool = self.gdf[self.gdf["isSelectedGrid"] == True]
        candidates = pool.index.values
        print(f"buildConflictGraph: found {len(existing)=} stations and {len(candidates)=} candidates")

        # map candidate grids to methods that selected them
        rankings = [f"$rank{m}" for m in self.methods]
        mappingCM = {}
        for idx, row in pool.iterrows():
            methods = {m for m, c in zip(self.methods, rankings) if row[c] > 0}
            if methods:
                mappingCM[idx] = methods

        # filter candidates too close to existing stations
        bt_existing = BallTree(self.gridCentroids[existing], leaf_size=leaf, metric="euclidean")
        distances, _ = bt_existing.query(self.gridCentroids[candidates], k=1)
        mask = distances.flatten() >= self.r_buffer
        candidates = candidates[mask]  # candidates far enough from existing stations
        print(f"buildConflictGraph: {len(candidates)=} candidates remain after buffer filtering")

        # indices of existing stations (gdf space)
        self.idxExisting: np.ndarray = existing
        # indices of candidate locations after proximity filtering (gdf space)
        self.bcgCandidates: np.ndarray = candidates
        # mapping of candidate indices to methods that selected them
        self.bcgMethodMaps = {idx: mappingCM[idx] for idx in candidates}

    def runClustering(self) -> None:
        centroids = self.gridCentroids[self.bcgCandidates]
        dbscan = DBSCAN(eps=self.d_epsilon, min_samples=self.min_candidates, metric="euclidean")
        clabels = dbscan.fit_predict(centroids)

        n_noise = list(clabels).count(-1)
        n_clusters = len(set(clabels)) - (1 if n_noise else 0)
        print(f"runClustering: DBSCAN found {n_clusters} clusters with {n_noise} noise points")

        # rank clusters by method diversity and size
        deets = []
        for c_rank in set(clabels):
            if c_rank == -1:
                continue  # skip noise

            c_mask = clabels == c_rank
            idxs_cluster = self.bcgCandidates[c_mask]

            # count unique methods in this cluster
            c_methods = set()
            for idx in idxs_cluster:
                c_methods.update(self.bcgMethodMaps[idx])

            deets.append(
                {
                    "rank": int(c_rank),
                    "idxs_cluster": idxs_cluster.tolist(),
                    "sz_cluster": len(idxs_cluster),
                    "methods": c_methods,
                    "n_methods": len(c_methods),
                }
            )

        # rank by method diversity (descending), then by size (descending)
        deets.sort(key=lambda x: (-x["n_methods"], -x["sz_cluster"]))

        # update ranks after sorting to reflect actual position
        for i, d in enumerate(deets):
            d["rank"] = i

        self.clusterLabels = clabels  # cluster labels for each candidate
        self.rankedClusters = deets  # list of cluster details, ranked

    def getMedoidGrid(self, idxs_cluster: np.ndarray) -> int:
        """finds the within-cluster medoid grid as representative of a cluster"""
        if len(idxs_cluster) == 1:
            return idxs_cluster[0]

        # compute pairwise distances
        c_centroids = self.gridCentroids[idxs_cluster]
        distances = cdist(c_centroids, c_centroids, metric="euclidean")

        # medoid is the point with minimum sum of distances to all other points
        dists = distances.sum(axis=1)
        idx_medoid = np.argmin(dists)

        return idxs_cluster[idx_medoid]

    def getCentroidGrid(self, idxs_cluster: np.ndarray) -> int:
        """finds the grid in the study area closest to geometric centroid of a cluster"""
        if len(idxs_cluster) == 1:
            return idxs_cluster[0]

        # compute geometric mean
        c_centroids = self.gridCentroids[idxs_cluster]
        mean_centroid = c_centroids.mean(axis=0)

        # find closest grid to mean
        dists = np.linalg.norm(c_centroids - mean_centroid, axis=1)
        idx_closest = np.argmin(dists)

        return idxs_cluster[idx_closest]

    def getExtensions(self) -> List[int]:
        """selects top-n clusters and returns list of representative grids from each cluster"""
        self.setup()
        self.buildConflictGraph()
        self.runClustering()

        # select top-n clusters
        topn = self.rankedClusters[: self.n]
        if len(topn) < self.n:
            print(f"getExtensions: only {len(topn)} clusters available, requested {self.n}")

        # select representative grid from each cluster
        self.idxSelected = []
        self.selectedClusters = []

        for cluster in topn:
            ci = cluster["idxs_cluster"]  # cluster indices in gdf space
            cri = self.getMedoidGrid(ci) if self.use_medoid else self.getCentroidGrid(ci)

            cri = int(cri)
            cluster["idx_selected"] = cri
            self.idxSelected.append(cri)  # cluster representative index in gdf space
            self.selectedClusters.append(cluster)

        self.idxSelected = np.array(self.idxSelected)
        assert self.validateSelection(), "!!!"
        print(f"getExtensions: selected {len(self.idxSelected)} extension stations")
        return self.gdf.loc[self.idxSelected, "id"].tolist()

    def validateSelection(self) -> bool:
        """validates selected stations are not within buffer radius of existing stations or each other"""
        reference = self.gdf.loc[self.idxExisting].copy()
        selected = self.gdf.loc[self.idxSelected].copy()
        g_reference = np.array([p.coords[0] for p in reference.geometry.centroid])
        g_selected = np.array([p.coords[0] for p in selected.geometry.centroid])
        isValid = True

        # check distance to existing stations
        t_reference = BallTree(g_reference, metric="euclidean")
        q_distances, q_indices = t_reference.query(g_selected, k=1)
        for i, dist in enumerate(q_distances):
            if dist[0] < self.r_buffer:
                print(f"station {i} is too close to existing stations ({dist[0]:.2f}m)")
                isValid = False

        # check distance between selected stations
        if len(g_selected) > 1:
            t_selected = BallTree(g_selected, metric="euclidean")
            q_distances, q_indices = t_selected.query(g_selected, k=2)

            skip = []
            for i, (dist, idx) in enumerate(zip(q_distances[:, 1], q_indices[:, 1])):
                if (dist < self.r_buffer) and ([idx, i] not in skip):
                    skip.append([i, idx])
                    print(f"stations {i} and {idx} are too close to each other ({dist:.2f}m)")
                    isValid = False

        print(f"validateSelection: validation check completed, {isValid=}")
        return isValid

    def plotExtensions(self, fpath: str | None = None) -> None:
        msg = "plotExtensions() must be implemented in subclass to visualise selected extensions"
        raise NotImplementedError(msg)


class aeSimilarityExpansion(extendExistingStations):
    """
    consensus-based station allocation for multiple AE configurations and parameterisations;
    requires `isSelectedGrid` and `$rank{method}` columns in the grid gdf for each selection method.
    """

    def setup(self) -> None:
        # run checks on gdf and methods
        assert "isBikeStation" in self.gdf.columns, "GeoDataFrame must contain 'isBikeStation'."

        for m in self.methods:
            assert f"$rank{m}" in self.gdf.columns, f"GeoDataFrame must contain '$rank{m}'."

        # precompute grid centroids for distance calculations
        centroids = self.gdf.geometry.centroid
        self.gridCentroids = np.vstack([centroids.x, centroids.y]).T

    def plotExtensions(self, fastmode=False, save: str | None = None):
        """visualises selected extensions with cluster information and method diversity"""
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        ax.set_axis_off()
        ax.margins(0.01)  # remove whitespace

        tab10, dark2 = plt.get_cmap("tab10"), plt.get_cmap("Dark2")
        cmap = list(tab10.colors) + list(dark2.colors)[::-1]  # type:ignore

        # plot study area
        border = gpd.GeoSeries(self.gdf.union_all(method="unary"))
        border.boundary.plot(ax=ax, color="black", linewidth=2)
        if not fastmode:
            self.gdf.plot(ax=ax, linewidth=0.1, edgecolor="lightgrey", facecolor="none")  # full grid

        # plot existing stations
        existing = self.gdf.loc[self.idxExisting].geometry.centroid
        existing.plot(ax=ax, color=cmap[0], marker="o", markersize=5, label="Existing Stations")

        # plot selected representatives
        selected = self.gdf.loc[self.idxSelected].geometry.centroid
        selected.plot(ax=ax, color=cmap[1], marker="s", markersize=21, label="Cluster Medoids", zorder=3)

        # plot all candidates by cluster
        for c in self.selectedClusters[: self.n]:
            cluster = self.gdf.loc[c["idxs_cluster"]]
            cluster.plot(ax=ax, color="green", alpha=0.65, linewidth=0.25)

            # plot cluster boundaries with some buffering
            c_hull = gpd.GeoSeries(cluster.union_all(method="unary").convex_hull.buffer(35))
            c_hull.boundary.plot(ax=ax, color="black", alpha=1.0, linewidth=0.5, zorder=4)

        # dummy plots for legend
        plt.scatter([], [], color="green", label=f"Cluster Candidates", s=9, marker="s")
        plt.plot([], [], color="black", linewidth=0.5, label="Cluster Boundaries")

        plt.legend(fontsize=10, loc="lower right", bbox_to_anchor=(0.98, 0.17))
        plt.tight_layout(pad=0)

        if save:
            plt.savefig(save, bbox_inches="tight")
            print(f"plotExtensions: figure saved to ./{save}")
            plt.close()
        return fig  # return figure for webapp display
