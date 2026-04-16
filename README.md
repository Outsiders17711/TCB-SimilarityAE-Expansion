# Similarity-Based Bike Station Expansion via Hybrid Denoising Autoencoders

This repository accompanies the conference paper *Similarity-Based Bike Station Expansion via Hybrid Denoising Autoencoders*. It contains the packaged study data, the analysis notebooks, and a Streamlit webapp for exploring the station allocation framework.

The deployed webapp is available at [tsae-webapp.streamlit.app/](https://tsae-webapp.streamlit.app/).

## Overview

The paper studies bike-sharing station expansion as a similarity-based location-allocation problem. Existing station locations are treated as reference examples, while candidate grid cells are ranked by how closely they resemble those references in feature space. The feature representation combines socio-demographic, built environment, transport network, and neighbourhood flow variables over a 100 by 100 metre grid covering Trondheim, Norway.

The central modelling step is a hybrid denoising autoencoder (HDAE) that compresses 29 engineered features into a structured latent space. Similarity is then computed in either the raw feature space or the learned embedding space, and candidate stations are selected with a greedy maximum-weight independent set heuristic under a 250 metre proximity constraint. The paper compares feature representations, similarity methods, and distance metrics, then derives consensus extension zones from repeated runs across multiple parametrisations.

## Repository Scope

The starting point is the packaged dataset in `data/tcbGridFeatures-TSAE29.gpkg` together with the prepackaged CSV exports in `data/`. The main files and directories are:

- `data/` contains the packaged grid dataset and precomputed cosine and Euclidean encodings used by the webapp.
- `data.ipynb` carries out exploratory analysis of the engineered spatial feature set.
- `model.ipynb` trains or reloads the HDAE, evaluates the learned embeddings, and exports the packaged artefacts used by the webapp.
- `webapp.py` provides the interactive dashboard for exploring the station allocation framework across different parameter settings.

Running the notebooks in order will also regenerate the prepackaged CSV artefacts under `data/` and produce visualisations under `imgs.bak/` and `models/`. The latter are not tracked in the repository but can be inspected locally.

## Environment Setup

The project uses [uv](https://docs.astral.sh/uv/) for environment management. To set up the environment with the required dependencies, run:

```bash
uv sync --frozen
```

This creates a virtual environment based on the `pyproject.toml` and `uv.lock` files.

## Run The Webapp

The webapp reads the packaged artefacts in `data/` and supporting information in `schema/`. You do not need to retrain the autoencoder or re-run the notebooks before launching it.

```bash
uv run streamlit run webapp.py
```

The dashboard exposes six analysis views that mirror the paper.

- Single Configuration
- Compare Features
- Compare Methods
- Compare Metrics
- TopK Sensitivity
- Consensus Selection

These views let you inspect the effects of feature representation, similarity aggregation, distance metric, and consensus filtering on the selected station candidates.
