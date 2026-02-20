from src.basics import *
from src.modules import *

# ------------------ globals ------------------ #
tab10, dark2 = plt.get_cmap("tab10"), plt.get_cmap("Dark2")
cmap = list(tab10.colors) + list(dark2.colors)[::-1]  # type:ignore

lfText = lambda params, t=[]: " | ".join([f"{k}={v}" for k, v in params.items() if k not in t])
nExisting = 68  # similarity is wrt to existing stations; some parameters are bounded by their count
root = Path(__file__).resolve().parent  # root directory of the webapp


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


# ------------------ allocation functions ------------------ #
def getSA(ds, features, params):
    with pipeCellOutput():
        AESA = aeStationAllocation(ds, features, **params)
        return AESA.runAllocation()


def plotMultiSelection(d_selected):
    data, border, existing = loadGridData()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_axis_off()
    ax.margins(0.01)

    # plot base layers and existing stations
    border.plot(ax=ax, color="black", linewidth=2)
    data.plot(ax=ax, linewidth=0.1, edgecolor="lightgrey", facecolor="none")
    existing.plot(ax=ax, color=cmap[0], label=f"Existing ({len(existing)})", marker="o", markersize=5)

    # plot overlap selections
    d_selected = {k: set(v) for k, v in d_selected.items()}
    s_overlap = set()
    if len(d_selected) > 1:
        s_overlap = set.intersection(*d_selected.values())
        if not s_overlap:  # print info and add empty legend entry
            print(f"INFO: empty overlapping set for <{'/'.join(d_selected.keys())}>")
            ax.scatter([], [], color=cmap[1], label="Overlap (0)", marker="s", s=24)
        else:
            gc = data[data["id"].isin(s_overlap)].geometry.centroid
            gc.plot(ax=ax, color=cmap[1], label=f"Overlap ({len(gc)})", marker="s", markersize=24)

    # plot individual selections
    for i, (label, s_selected) in enumerate(d_selected.items(), start=2):
        ids = s_selected - s_overlap
        if not ids:  # print info and add empty legend entry
            print(f"INFO: empty non-overlapping set for <{label}>")
            ax.scatter([], [], color=cmap[i], label=f"{label} (0)", marker="s", s=11)
        else:
            gc = data[data["id"].isin(ids)].geometry.centroid
            gc.plot(ax=ax, color=cmap[i], label=label, marker="s", markersize=11)

    plt.legend(fontsize=10, loc="lower right", bbox_to_anchor=(0.98, 0.17))
    plt.tight_layout(pad=0)
    return fig


def plotMultiLevelOverlap(d_selected):
    data, border, existing = loadGridData()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_axis_off()
    ax.margins(0.01)

    # plot base layers and existing stations
    border.plot(ax=ax, color="black", linewidth=2)
    data.plot(ax=ax, linewidth=0.1, edgecolor="lightgrey", facecolor="none")
    existing.plot(ax=ax, color=cmap[0], label=f"Existing ({len(existing)})", marker="o", markersize=5)

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
            print(f"INFO: empty set for <{label}> threshold")
            ax.scatter([], [], color=cmap[i], label=f"{label} (0)", marker=marker, s=size)

    # plot unique selections
    s_overlap = set.union(*l_thresholds.values()) if l_thresholds else set()
    for j, (label, s_selected) in enumerate(d_selected.items(), start=len(l_thresholds) + 1):
        unique = s_selected - s_overlap
        if unique:
            gc = data[data["id"].isin(unique)].geometry.centroid
            gc.plot(ax=ax, color=cmap[j], label=f"{label} ({len(gc)})", marker="s", markersize=9)
        else:  # print info and add empty legend entry
            print(f"INFO: empty unique set for <{label}>")
            ax.scatter([], [], color=cmap[j], label=f"{label} (0)", marker="s", s=9)

    plt.legend(fontsize=10, loc="lower right", bbox_to_anchor=(0.98, 0.17))
    plt.tight_layout(pad=0)
    return fig


def getDatasetByFeature(datasets, feature, metric):
    """helper to select dataset and features based on user selection"""
    if feature == "AE Encodings":
        ds = datasets[metric]
        features = datasets["f_encodings"]
    else:
        ds = datasets["raw"]
        features = datasets["f_exogenous"]
    return ds, features


def chooseK():
    strategy = st.sidebar.radio(
        "**K Selection Strategy**",
        ["Logarithmic", "Percentile", "Manual"],
        help="Logarithmic: spread K values logarithmically; Percentile: select K values based on percentiles of the range; Manual: specify exact K values as comma-separated list",
    )

    if strategy == "Manual":
        csv_k = st.sidebar.text_input("**K Values (comma-separated)**", "1,5,10,20,35,50,68")
        return [int(k.strip()) for k in csv_k.split(",")]

    n_values = st.sidebar.slider("Number of K values", min_value=3, max_value=12, value=7)
    lf_uci = lambda values: np.unique(np.ceil(values).astype(int))

    if strategy == "Logarithmic":
        values = lf_uci(np.logspace(0, np.log10(nExisting), n_values))
    else:  # percentile
        values = lf_uci(np.percentile(range(1, nExisting + 1), np.linspace(0, 100, n_values)))

    values[0], values[-1] = 1, nExisting  # pin endpoints
    assert len(values) == n_values, "duplicate K values generated; adjust number/strategy parameters"
    return values.tolist()


# ------------------ streamlit app ------------------ #
def stPageConfig():
    st.set_page_config(
        page_title="TCB SimilarityAE Expansion (TSAE) Webapp",
        layout="centered",
        initial_sidebar_state=360,  # "auto" behavior but start with the specified width
    )

    # custom css to reduce sidebar spacing
    dtid = "[data-testid='stSidebar']"
    st.markdown(
        f"""
        <style>
        /* sidebar padding */
        {dtid} {{ padding-top: 0; padding-bottom: 0; }}
        /* spacing around markdown elements in sidebar */
        {dtid} .element-container {{ margin-bottom: 0rem; }}
        /* spacing around horizontal rules */
        {dtid} hr {{ margin-top: 0rem; margin-bottom: 0.5rem; }}
        /* title spacing */
        {dtid} h1, {dtid} h2, {dtid} h3 {{ padding-top: 0; margin-top: 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def landing():
    """Display landing page with app description and parameter information"""
    st.title("🚲 TSAE Similarity-Based BSS Expansion Dashboard")
    st.markdown("""
    This dashboard enables interactive comparison of bike station allocation strategies 
    across different parametrisations using autoencoder-based spatial modelling.
    """)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Analysis Modes")
        st.markdown("""
        - **Single Configuration**: Test individual parameter combinations
        - **Compare Features**: AE encodings vs raw spatial features
        - **Compare Methods**: TopK vs KDE allocation methods
        - **Compare Metrics**: Cosine vs Euclidean distance
        - **TopK Sensitivity**: Analyse robustness across K values
        - **Consensus Selection**: Spatial clustering for robust placement
        """)
    with col2:
        st.markdown("#### ⚙️ Key Parameters")
        st.markdown("""
        - **Features**: AE encodings (learned representations) or raw spatial features
        - **Metric**: Distance measure (cosine for l2-normalised, euclidean otherwise)
        - **Method**: TopK (nearest neighbors) or KDE (kernel density)
        - **TopK**: Number of nearest existing stations to consider
        - **nStations**: Target number of new stations to allocate
        """)

    st.markdown("---")
    st.info("👈 Select an analysis mode from the sidebar to begin")


def main():
    stPageConfig()

    # initialise session state for caching results
    if "results" not in st.session_state:
        st.session_state.results = {}

    # load data
    with st.spinner("Loading data and models..."):
        datasets = prepareDatasets()

    # sidebar navigation
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    home = col1.button("**🏠 TSAE Dashboard**", type="secondary", width="stretch")
    reload = col2.button("🔄", help="Reload page and clear cache", type="secondary")

    st.sidebar.markdown("### Select Analysis Mode")
    analysis = st.sidebar.radio(
        "**Analysis Mode**",
        [
            "Single Configuration",
            "Compare Features",
            "Compare Methods",
            "Compare Metrics",
            "TopK Sensitivity",
            "Consensus Selection",
        ],
        index=None,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    if reload:
        st.session_state.results = {}
        st.rerun()

    if home or (analysis is None):
        landing()
    elif analysis == "Single Configuration":
        runSingleConfig(datasets)
    elif analysis == "Compare Features":
        runFeatureComparison(datasets)
    elif analysis == "Compare Methods":
        runMethodComparison(datasets)
    elif analysis == "Compare Metrics":
        runMetricComparison(datasets)
    elif analysis == "TopK Sensitivity":
        runTopKSensitivity(datasets)
    elif analysis == "Consensus Selection":
        runConsensusSelection(datasets)


def selectParams(page: str):
    suffixes = {"single": "Configuration", "topk": "Sensitivity", "consensus": "Synthesis"}
    suffix = suffixes.get(page, "Comparison")
    st.header(f"{page.title()} {suffix}", divider="gray")

    st.sidebar.markdown("### Configuration")
    n = st.sidebar.number_input("**Number of Stations (n)**", 12, max_value=204, value=nExisting)

    feature = "AE Encodings"  # default
    if page not in ["feature", "metric"]:
        feature = st.sidebar.radio("**Features**", ["AE Encodings", "Raw Features"], horizontal=True)

    metric = "cosine"  # default
    if page != "metric":
        metric = st.sidebar.radio("**Metric**", ["cosine", "euclidean"], horizontal=True)

    method = "topk"  # default
    if page not in ["method", "topk", "consensus"]:
        method = st.sidebar.radio("**Method**", ["topk", "kde"], horizontal=True)

    topk = None  # default
    if (page not in ["topk", "consensus"]) and (method == "topk"):
        topk = st.sidebar.slider("Top K", min_value=1, max_value=nExisting, value=3)
    elif page in ["topk", "consensus"]:
        topk = chooseK()
        # st.sidebar.write(f"K values: {topk}")  # already displayed in main area

    clicked = False
    if page != "consensus":
        clicked = st.sidebar.button(f"(re)Run {suffix}", type="primary", width="stretch")

    cached = page in st.session_state.results  # show cached results if available
    settings = {"n": n, "feature": feature, "method": method, "metric": metric, "topk": topk}
    return settings, clicked, cached


def showResults(page, params, p_excludes=[]):
    """helper to display results with consistent formatting"""
    result = st.session_state.results[page]
    st.markdown(f"**Parameters:** {lfText(params, [page, *p_excludes])}")
    st.pyplot(fig=result["fig"])

    if "stats" in result:
        cols = len(result["stats"])
        for col, (stat, value) in zip(st.columns(cols), result["stats"].items()):
            col.metric(stat, value)

    mapping = {"single": "selected", "consensus": "consensus"}
    tag = mapping.get(page, "overlap")
    with st.expander(f"View {tag.title()} Grid IDs"):
        if tag == "consensus":
            df = pd.DataFrame(result["deetsConsensus"])
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.write(sorted(result[tag]))


def runSingleConfig(datasets):
    page = "single"
    settings, clicked, cached = selectParams(page)
    n, feature, method, metric, topk = settings.values()
    params = {"n": n, "metric": metric, "method": method, "topk": topk}

    if clicked or (not cached):
        with st.spinner("Running allocation..."):
            ds, features = getDatasetByFeature(datasets, feature, metric)
            s_selected = getSA(ds, features, params)
            d_selected = {feature: s_selected}

            fig = plotMultiSelection(d_selected)
            st.session_state.results[page] = {"selected": s_selected, "fig": fig}

    showResults(page, params)


def runFeatureComparison(datasets):
    page = "feature"
    settings, clicked, cached = selectParams(page)
    n, _, method, metric, topk = settings.values()
    params = {"n": n, "metric": metric, "method": method, "topk": topk}

    if clicked or (not cached):
        with st.spinner("Running allocations x2..."):
            dsAE = datasets[metric]
            saAE = getSA(dsAE, datasets["f_encodings"], params)
            saRaw = getSA(datasets["raw"], datasets["f_exogenous"], params)

            d_selected = {"AE Encodings": saAE, "Raw Features": saRaw}
            overlap = set(saAE) & set(saRaw)
            pct = len(overlap) / n * 100

            fig = plotMultiSelection(d_selected)
            st.session_state.results[page] = {
                "fig": fig,
                "overlap": overlap,
                "stats": {"100% Overlap": f"{len(overlap)} stations", "Overlap %": f"{pct:.2f}%"},
            }

    showResults(page, params)


def runMethodComparison(datasets):
    page = "method"
    settings, clicked, cached = selectParams(page)
    n, feature, _, metric, topk = settings.values()
    params = {"n": n, "metric": metric, "method": None, "topk": topk}

    if clicked or not (cached):
        with st.spinner("Running allocations x2..."):
            ds, features = getDatasetByFeature(datasets, feature, metric)
            d_selected = {}
            for method in ["topk", "kde"]:  # run both methods
                params["method"] = method
                d_selected[method.upper()] = getSA(ds, features, params)

            overlap = set(d_selected["TOPK"]) & set(d_selected["KDE"])
            pct = len(overlap) / n * 100

            fig = plotMultiSelection(d_selected)
            st.session_state.results[page] = {
                "fig": fig,
                "overlap": overlap,
                "stats": {"100% Overlap": f"{len(overlap)} stations", "Overlap %": f"{pct:.2f}%"},
            }

    showResults(page, params)


def runMetricComparison(datasets):
    page = "metric"
    settings, clicked, cached = selectParams(page)
    n, feature, method, _, topk = settings.values()
    params = {"n": n, "metric": None, "method": method, "topk": topk}

    if clicked or not (cached):
        with st.spinner("Running allocations..."):
            d_selected = {}
            for metric in ["cosine", "euclidean"]:  # run both metrics
                params["metric"] = metric
                ds, features = getDatasetByFeature(datasets, feature, metric)
                d_selected[metric.capitalize()] = getSA(ds, features, params)

            overlap = set(d_selected["Cosine"]) & set(d_selected["Euclidean"])
            pct = len(overlap) / n * 100

            fig = plotMultiSelection(d_selected)
            st.session_state.results[page] = {
                "fig": fig,
                "overlap": overlap,
                "stats": {"100% Overlap": f"{len(overlap)} stations", "Overlap %": f"{pct:.2f}%"},
            }

    showResults(page, params)


def runTopKSensitivity(datasets):
    page = "topk"
    settings, clicked, cached = selectParams(page)
    n, feature, method, metric, k_values = settings.values()
    params = {"n": n, "metric": metric, "method": method, "topk": None}

    if clicked or not (cached):
        with st.spinner("Running allocations..."):
            ds, features = getDatasetByFeature(datasets, feature, metric)
            d_selected = {}
            for k in k_values:
                params["topk"] = k
                d_selected[f"K={k}"] = getSA(ds, features, params)

            l_selections = [set(s) for s in d_selected.values()]
            overlap = set.intersection(*l_selections)
            selected = set.union(*l_selections)

            fig = plotMultiLevelOverlap(d_selected)
            st.session_state.results[page] = {
                "fig": fig,
                "overlap": overlap,
                "selected": selected,
                "stats": {
                    "100% Overlap": f"{len(overlap)} stations",
                    "Selected": f"{len(selected)} stations",
                    "Unique Ratio": f"{len(selected) / n:.2f}x",
                },
            }

    st.markdown(f"**K Values:** {', '.join(map(str, k_values))}")
    showResults(page, params)


def runConsensusSelection(datasets):
    page = "consensus"
    settings, _, cached = selectParams(page)
    n, feature, method, metric, k_values = settings.values()
    params = {"n": n, "metric": metric, "method": method, "topk": None}

    st.sidebar.markdown("### Spatial Clustering")
    threshold = st.sidebar.slider(
        "**Consensus Threshold (%)**",
        min_value=25,
        max_value=100,
        value=100,
        step=25,
        help="Minimum percentage of methods that must select a station",
    )

    col1, col2 = st.sidebar.columns(2)
    kwargs = {"min_value": 250, "max_value": 1000, "value": 250, "step": 50}
    r_buffer = col1.number_input(
        label="**Buffer Radius (m)**",
        help="Minimum spatial separation between candidate and existing stations",
        **kwargs,  # type:ignore
    )
    d_epsilon = col2.number_input(
        "**DBSCAN Epsilon (m)**",
        help="Maximum distance for points to be considered in the same cluster",
        **kwargs,  # type:ignore
    )
    sz_cluster = st.sidebar.number_input("**Minimum Cluster Size**", min_value=1, max_value=10, value=2)
    use_medoid = st.sidebar.checkbox("**Use Medoid for Selection**", value=True)

    clicked = st.sidebar.button("Run Consensus Analysis", type="primary", width="stretch")
    if clicked or not (cached):
        with st.spinner("Running consensus synthesis..."):
            ds, features = getDatasetByFeature(datasets, feature, metric)
            d_selected = {}
            for k in k_values:
                params["topk"] = k
                d_selected[f"K={k}"] = getSA(ds, features, params)

            # create grid with selections
            gdf, *be = loadGridData()
            gdf = gdf[["id", "isBikeStation", "geometry"]].copy()
            gdf["isSelectedGrid"] = 0

            # mark selected stations for each method
            for k, s_selected in d_selected.items():
                gdf.loc[gdf["id"].isin(s_selected), "isSelectedGrid"] = 1
                k = int(k.split("=")[1])
                c = f"$rankK{k:02}"
                gdf[c] = np.where(gdf["id"].isin(s_selected), k, 0)
            methods = [c.replace("$rank", "") for c in gdf.columns if c.startswith("$rank")]

            # run consensus spatial clustering
            with pipeCellOutput():
                ASE = aeSimilarityExpansion(
                    gdf,
                    n=n,
                    s_methods=methods,
                    r_buffer=r_buffer,
                    d_epsilon=d_epsilon,
                    min_candidates=sz_cluster,
                    use_medoid=use_medoid,
                )
                ge = ASE.getExtensions()

            # filter by consensus threshold
            min_methods = int(np.ceil(threshold / 100 * len(methods)))
            filteredClusters = [c for c in ASE.selectedClusters if c["n_methods"] >= min_methods]
            ASE.selectedClusters = filteredClusters
            ASE.idxSelected = [c["idx_selected"] for c in filteredClusters]

            # prepare consensus data
            deetsConsensus = []
            for c in filteredClusters:
                deetsConsensus.append(
                    {
                        "Grid ID": gdf.loc[c["idx_selected"], "id"],
                        "Cluster Size": c["sz_cluster"],
                        "Methods": c["n_methods"],
                        "Consensus %": f"{c['n_methods'] / len(methods) * 100:.1f}%",
                    }
                )

            fig = ASE.plotExtensions()
            st.session_state.results[page] = {
                "fig": fig,
                "filteredClusters": filteredClusters,
                "rankedClusters": ASE.rankedClusters,
                "deetsConsensus": deetsConsensus,
                "stats": {
                    "Total Clusters": len(ASE.rankedClusters),
                    "Consensus Clusters": len(filteredClusters),
                    "Parametrisations Tested": len(methods),
                },
            }

    st.markdown(f"**K Values:** {', '.join(map(str, k_values))}")
    showResults(page, params, p_excludes=["topk"])


if __name__ == "__main__":
    main()
