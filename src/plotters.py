import io

from .basics import *
from .modules import *

__all__ = [
    "loadGridData",
    "prepareDatasets",
    "buildBaseFig",
    "plotMultiSelection",
    "plotMultiLevelOverlap",
    "plotConsensusExtensions",
]

root = Path(__file__).resolve().parent.parent  # root directory of the webapp
tab10, dark2 = plt.get_cmap("tab10"), plt.get_cmap("Dark2")
cmap = list(tab10.colors) + list(dark2.colors)[::-1]  # type:ignore


# ------------------ data loading ------------------ #
@st.cache_resource
def loadGridData():
    data = csv2gdf(f"{root}/data/tcbGridFeatures-TSAE29.csv")
    border = gpd.GeoSeries(data.union_all(method="unary")).boundary
    existing = data[data["isBikeStation"] == True].geometry.centroid
    return data, border, existing


@st.cache_data
def prepareDatasets():
    f_grid = ["id", "isBikeStation", "geometry"]

    # load processed (zscored) spatial features from tabular object (`to.items`)
    raw = csv2gdf(f"{root}/data/TO-SpatialFeatures.csv")
    f_exogenous = [c for c in raw.columns if c not in f_grid]

    # load separate versions of trained encodings for euclidean and cosine metrics
    euclidean = csv2gdf(f"{root}/data/AE-EuclideanEncodings.csv")
    cosine = csv2gdf(f"{root}/data/AE-CosineEncodings.csv")
    f_encodings = [c for c in cosine.columns if c not in f_grid]

    return {
        "cosine": cosine,
        "euclidean": euclidean,
        "raw": raw,
        "f_encodings": f_encodings,
        "f_exogenous": f_exogenous,
    }


@st.cache_resource
def cacheBasePlot():
    """renders base layers once and returns cached PNG bytes with coordinate extents"""
    data, border, existing = loadGridData()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_axis_off()
    ax.margins(0.01)

    border.plot(ax=ax, color="black", linewidth=2)
    data.plot(ax=ax, linewidth=0.1, edgecolor="lightgrey", facecolor="none")
    existing.plot(ax=ax, color=cmap[0], marker="o", markersize=5)
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore
    bbox = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    layers = io.BytesIO()
    fig.savefig(layers, format="png", dpi=150, bbox_inches=bbox)
    plt.close(fig)

    return layers.getvalue(), xlim, ylim, len(existing)


# ------------------ plotting functions ------------------ #
def buildBaseFig():
    """returns a new fig/ax with base layers pre-rendered as a cached background image"""
    layers, xlim, ylim, n_existing = cacheBasePlot()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_axis_off()
    img = plt.imread(io.BytesIO(layers))

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    ax.imshow(img, extent=extent, aspect="equal", origin="upper", zorder=0)  # type:ignore
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.scatter([], [], color=cmap[0], label=f"Existing ({n_existing})", marker="o", s=25)

    return fig, ax


def plotMultiSelection(d_selected):
    data, *be = loadGridData()
    fig, ax = buildBaseFig()

    # plot overlap selections
    d_selected = {k: set(v) for k, v in d_selected.items()}
    s_overlap = set()
    if len(d_selected) > 1:
        s_overlap = set.intersection(*d_selected.values())
        if not s_overlap:  # print info and add empty legend entry
            logInfo(f"empty overlapping set for <{'/'.join(d_selected.keys())}>")
            ax.scatter([], [], color=cmap[1], label="Overlap (0)", marker="s", s=24)
        else:
            gc = data[data["id"].isin(s_overlap)].geometry.centroid
            gc.plot(ax=ax, color=cmap[1], label=f"Overlap ({len(gc)})", marker="s", markersize=24)

    # plot individual selections
    start = 2 if s_overlap else 1
    for i, (label, s_selected) in enumerate(d_selected.items(), start=start):
        ids = s_selected - s_overlap
        if not ids:  # print info and add empty legend entry
            logInfo(f"empty non-overlapping set for <{label}>")
            ax.scatter([], [], color=cmap[i], label=f"{label} (0)", marker="s", s=11)
        else:
            gc = data[data["id"].isin(ids)].geometry.centroid
            gc.plot(ax=ax, color=cmap[i], label=label, marker="s", markersize=11)

    plt.legend(fontsize=10, loc="lower right", bbox_to_anchor=(0.98, 0.17))
    plt.tight_layout(pad=0)
    return fig


def plotMultiLevelOverlap(d_selected):
    data, *be = loadGridData()
    fig, ax = buildBaseFig()

    # count occurrences
    d_selected = {k: set(v) for k, v in d_selected.items()}
    n_sets = len(d_selected)
    counter = Counter([id for subset in d_selected.values() for id in subset])

    # define threshold levels
    thresholds = {
        "Overlap": (1.0, "s", 24),
        "75%+ Overlap": (0.75, "^", 21),
        "50%+ Overlap": (0.50, "D", 15),
    }
    l_thresholds = {}
    for label, (t, m, ms) in thresholds.items():
        min_count = int(np.ceil(t * n_sets))
        l_thresholds[label] = {id for id, count in counter.items() if count >= min_count}

    # plot overlap levels
    plotted = set()
    for i, (label, s_selected) in enumerate(l_thresholds.items(), start=1):
        level = s_selected - plotted
        marker, size = thresholds[label][1:]
        if level:
            gc = data[data["id"].isin(level)].geometry.centroid
            gc.plot(ax=ax, color=cmap[i], label=f"{label} ({len(gc)})", marker=marker, markersize=size)
            plotted.update(level)
        else:  # print info and add empty legend entry
            logInfo(f"empty set for <{label}> threshold")
            ax.scatter([], [], color=cmap[i], label=f"{label} (0)", marker=marker, s=size)

    # plot unique selections
    s_overlap = set.union(*l_thresholds.values()) if l_thresholds else set()
    for j, (label, s_selected) in enumerate(d_selected.items(), start=len(l_thresholds) + 1):
        unique = s_selected - s_overlap
        if unique:
            gc = data[data["id"].isin(unique)].geometry.centroid
            gc.plot(ax=ax, color=cmap[j], label=f"{label} ({len(gc)})", marker="s", markersize=9)
        else:  # print info and add empty legend entry
            logInfo(f"empty unique set for <{label}>")
            ax.scatter([], [], color=cmap[j], label=f"{label} (0)", marker="s", s=9)

    plt.legend(fontsize=10, loc="lower right", bbox_to_anchor=(0.98, 0.17))
    plt.tight_layout(pad=0)
    return fig


def plotConsensusExtensions(idxSelected, selectedClusters, *, n):
    data, *be = loadGridData()
    fig, ax = buildBaseFig()

    # plot selected representatives
    selected = data.loc[idxSelected].geometry.centroid
    selected.plot(ax=ax, color=cmap[1], marker="s", markersize=21, label="Cluster Medoids", zorder=3)

    # plot all candidates by cluster
    for c in selectedClusters[:n]:
        cluster = data.loc[c["idxs_cluster"]]
        cluster.plot(ax=ax, color="green", alpha=0.65, linewidth=0.25)

        # plot cluster boundaries with some buffering
        c_hull = gpd.GeoSeries(cluster.union_all(method="unary").convex_hull.buffer(35))
        c_hull.boundary.plot(ax=ax, color="black", alpha=1.0, linewidth=0.5, zorder=4)

    # dummy plots for legend
    plt.scatter([], [], color="green", label=f"Cluster Candidates", s=9, marker="s")
    plt.plot([], [], color="black", linewidth=0.5, label="Cluster Boundaries")

    plt.legend(fontsize=10, loc="lower right", bbox_to_anchor=(0.98, 0.17))
    plt.tight_layout(pad=0)
    return fig
