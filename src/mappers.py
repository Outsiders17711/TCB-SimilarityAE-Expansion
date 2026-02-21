import matplotlib.colors as mcolors

from .basics import *
from .modules import *
from .plotters import cmap, loadGridData

__all__ = [
    "mapMultiSelection",
    "mapMultiLevelOverlap",
    "mapConsensusExtensions",
]


# ------------------ helpers ------------------ #
cmap = [mcolors.to_hex(c) for c in cmap]  # convert matplotlib colors to hex for folium


def addSwatch(label: str, color: str) -> str:
    swatch = f'<span style="display:inline-block; width:8px; height:8px; background:{color}; border:1.5px solid black; flex-shrink:0; margin:0 3px 0 3px;"></span>'
    return f"{swatch}{label}"


def getStyling(color, *, name=None, marker=None, size=None):
    styler = {
        "color": color,
        "style_kwds": {
            "weight": 1.5,
            "opacity": 1.0,
            "fill": True,
            "color": "black",
            "fillColor": color,
            "fillOpacity": 0.85,
        },
        # ---
        "marker_type": marker,
        "marker_kwds": {"radius": size, "fill": True, "icon": None},
        # ---
        "tooltip": False,
        "tooltip_kwds": {"style": "font-size: 1.15rem;"},
        "highlight": True,
    }
    if name is not None:
        styler["name"] = addSwatch(name, color)

    return styler


def initBaseMap(data: gpd.GeoDataFrame) -> folium.Map:
    c = data.union_all().centroid
    m = folium.Map(
        location=[c.y, c.x],
        zoom_start=12,
        tiles=None,
        zoom_control=False,
        control_scale=True,
    )
    alt = folium.TileLayer("cartodbdark_matter", name="CartoDB Dark Matter", control=True).add_to(m)
    ll = folium.TileLayer("cartodbpositron", name="CartoDB Positron", control=True).add_to(m)
    lr = folium.TileLayer("openstreetmap", name="OpenStreetMap", control=True).add_to(m)
    # fp.SideBySideLayers(ll, lr).add_to(m)
    fp.MousePosition().add_to(m)

    # custom CSS for layer control
    css = """
    <style>
    .leaflet-control-layers { font-size: 10px; line-height: 1.6; font-weight: bold; }
    .leaflet-control-layers-list label { display: flex; align-items: center; gap: 5px; }
    </style>
    """
    m.get_root().html.add_child(folium.Element(css))  # type:ignore

    # plot existing stations
    existing = data[data["isBikeStation"] == True].copy()
    existing.explore(m=m, **getStyling(cmap[0], name=f"Existing ({len(existing)})"))

    return m


def _pts(data: gpd.GeoDataFrame, ids: set) -> gpd.GeoDataFrame:
    gdf = data[data["id"].isin(ids)].copy()
    return gdf


# ------------------ folium plotters ------------------ #
def mapMultiSelection(d_selected: dict) -> folium.Map:
    data, *be = loadGridData()
    m = initBaseMap(data)

    # plot overlap selections
    d_selected = {k: set(v) for k, v in d_selected.items()}
    s_overlap = set()
    if len(d_selected) > 1:
        s_overlap = set.intersection(*d_selected.values())
        if not s_overlap:
            logInfo(f"empty overlapping set for <{'/'.join(d_selected.keys())}>")
        else:
            gc = data[data["id"].isin(s_overlap)]
            gc.explore(m=m, **getStyling(cmap[1], name=f"Overlap ({len(gc)})"))

    # plot individual selections
    start = 2 if s_overlap else 1
    for i, (label, s_selected) in enumerate(d_selected.items(), start=start):
        ids = s_selected - s_overlap
        if not ids:
            logInfo(f"empty non-overlapping set for <{label}>")
        else:
            gc = data[data["id"].isin(ids)]
            gc.explore(m=m, **getStyling(cmap[i], name=label))

    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)
    m.fit_bounds(m.get_bounds())  # type:ignore
    return m


def mapMultiLevelOverlap(d_selected: dict) -> folium.Map:
    data, *be = loadGridData()
    m = initBaseMap(data)

    # count occurrences
    d_selected = {k: set(v) for k, v in d_selected.items()}
    n_sets = len(d_selected)
    counter = Counter([id for subset in d_selected.values() for id in subset])

    # define threshold levels
    thresholds = {
        "Overlap": (1.0, "s", 9),
        "75%+ Overlap": (0.75, "^", 7),
        "50%+ Overlap": (0.50, "D", 5),
    }
    l_thresholds = {}
    for label, (t, *mms) in thresholds.items():
        min_count = int(np.ceil(t * n_sets))
        l_thresholds[label] = {id for id, count in counter.items() if count >= min_count}

    # plot overlap levels
    plotted = set()
    for i, (label, s_selected) in enumerate(l_thresholds.items(), start=1):
        level = s_selected - plotted
        marker, size = thresholds[label][1:]
        if level:
            gc = data[data["id"].isin(level)]
            gc.explore(m=m, **getStyling(cmap[i], name=f"{label} ({len(gc)})"))
            plotted.update(level)
        else:
            logInfo(f"empty set for <{label}>")

    # plot unique selections
    s_overlap = set.union(*l_thresholds.values()) if l_thresholds else set()
    for j, (label, s_selected) in enumerate(d_selected.items(), start=len(l_thresholds) + 1):
        unique = s_selected - s_overlap
        if unique:
            gc = data[data["id"].isin(unique)]
            gc.explore(m=m, **getStyling(cmap[j], name=f"{label} ({len(gc)})"), show=False)
        else:
            logInfo(f"empty unique set for <{label}>")

    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)
    m.fit_bounds(m.get_bounds())  # type:ignore
    return m


def mapConsensusExtensions(idxSelected, selectedClusters, *, n) -> folium.Map:
    data, *be = loadGridData()
    m = initBaseMap(data)

    # plot cluster boundaries with some buffering
    l_hulls = [
        data.loc[c["idxs_cluster"]].union_all(method="unary").convex_hull.buffer(35)
        for c in selectedClusters[:n]
    ]
    hulls = gpd.GeoDataFrame(geometry=gpd.GeoSeries(l_hulls, crs=data.crs))
    hkw = {"color": cmap[1], "style_kwds": {"weight": 1.25}, "fill": False}
    hulls.explore(m=m, name=addSwatch("Cluster Boundaries", cmap[1]), **hkw)

    # plot all candidates by cluster
    clusters = gpd.pd.concat([data.loc[c["idxs_cluster"]] for c in selectedClusters[:n]])
    clusters.explore(m=m, **getStyling("green", name=f"Cluster Candidates ({len(clusters)})"))

    # plot selected representatives
    selected = data.loc[idxSelected]
    selected.explore(m=m, **getStyling(cmap[1], name=f"Cluster Medoids ({len(selected)})"))

    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)
    m.fit_bounds(m.get_bounds())  # type:ignore
    return m
