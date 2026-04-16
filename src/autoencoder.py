from dataclasses import MISSING, fields

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .basics import *
from .modules import sanityCheck, selfRNDR

__all__ = [
    "configAE",
    "dataloaderAE",
    "baseAutoEncoder",
    "hybridAutoEncoder",
    "trainAE",
    "getEncodedFeatures",
    "featureExtractor",
    "trainUtils",
]


@dataclass
class configAE:
    # --- training parameters ---
    batch_size: int = 512
    learning_rate: float = 1e-3
    n_epochs: int = 1024
    early_stopping: bool = True
    es_patience: int = 15  # early stopping patience
    save_model: bool = False  # whether to save model checkpoints

    # --- model parameters ---
    n_features: int = 0  # [NOTE] actual value set dynamically
    n_layers: int = 2  # [NOTE] 2/compressed, 3/detailed
    hidden_dim: int = 16  # [NOTE] reference dimension for hidden layer scaling
    exp_scaling: bool = True  # use exponents (2**i) or multiples (i+1) for hidden layer scaling
    embed_dim: int = 8  # [NOTE] leave as is for good compression/representation
    dropout_pct: float = 0.0  # [NOTE] rely on noise augmentation for needed robustness
    tie_weights: bool = False  # [NOTE] handicaps encoder unnecessarily for target task

    # --- supervised classification ---
    supervised: bool = True  # whether to use hybrid autoencoder with classification head
    wf_alpha: int = 10  # weighting factor for classification loss to handle class imbalance
    wf_lambda: float = 0.1  # weighting factor for classification loss in hybrid loss
    ns_classes: int = 1  # boolean target (isBikeStation) for classification

    # --- data augmentation ---
    noise_augment: bool = True
    na_probability: float = 0.3  # probability of adding noise to each feature
    na_sigma: float = 0.1  # standard deviation of gaussian noise

    # --- misc parameters ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "bae"
    model_dir: str = "../models"
    timestamp: str = datetime.now().strftime("%d%m%H%M")
    metrics: str | None = None  # [NOTE] actual value set after training


class dataloaderAE:
    """
    simple dataloader for autoencoder training;
    requires `configAE` object for configuration and `TabularPandas` (`to`) object for data.
    """

    def __init__(self, to, cfg: configAE, **kwargs):
        self.to = to.copy()
        self.bs = cfg.batch_size
        self.device = cfg.device
        self.kwargs = kwargs

        self.setup()

    def setup(self):
        """sets up train and valid datasets from TabularPandas object"""

        # lambda functions to convert pandas DataFrame/Series to torch tensors
        xlf = lambda x: torch.tensor(x.values, dtype=torch.float32).to(self.device)
        ylf = lambda y: torch.tensor(y.values, dtype=torch.long).to(self.device)

        self.train_ds = TensorDataset(xlf(self.to.train.xs), ylf(self.to.train.ys))
        if self.to.valid.items.empty:  # use all data for training and validation
            print("warning: no validation set detected; using training set for validation")
            self.valid_ds = TensorDataset(xlf(self.to.train.xs), ylf(self.to.train.ys))
        else:
            self.valid_ds = TensorDataset(xlf(self.to.valid.xs), ylf(self.to.valid.ys))

    def dataloaders(self):
        """
        returns train and valid dataloaders; data has been shuffled once by `to` object at setup;
        here train dataloader also shuffles each epoch.
        """
        train_dls = DataLoader(self.train_ds, batch_size=self.bs, shuffle=True, **self.kwargs)
        valid_dls = DataLoader(self.valid_ds, batch_size=self.bs, shuffle=False, **self.kwargs)
        return train_dls, valid_dls


class augFeatureNoise(nn.Module):
    """
    simple feature noise augmentation layer for autoencoders;
    adds gaussian noise to a subset of features during training.
    """

    def __init__(self, p_feature: float = 0.3, sigma: float = 0.1, clip: list | None = [-3, 3]):
        """
        p_feature: probability of adding noise to each feature;
        sigma: standard deviation of gaussian noise in normalised space;
        clip: min/max values to clip noise to (in normalised space); set to `None` to disable clipping.
        """
        super().__init__()
        self.p_feature = p_feature
        self.sigma = sigma
        self.clip = clip

    def forward(self, x):
        if not self.training:
            return x  # no augmentation during evaluation

        noise = torch.randn_like(x) * self.sigma
        # (bernoulli) mask of which features to perturb per sample
        mask = torch.rand_like(x) < self.p_feature

        # [NOTE] two options for clipping:
        # 1. "never perturb a feature by more than sigma" -> clip the noise itself
        # 2. "stay within the model's training distribution" -> clip the noised features
        if self.clip is not None:
            noise = torch.clamp(noise, self.clip[0], self.clip[1])

        x = x + noise * mask
        return x


class baseAutoEncoder(nn.Module):
    """
    simple autoencoder architecture; reasonably deep and wide to capture complex relationships;
    forward method can return either the encoded representation or the full reconstruction.
    """

    def __init__(self, cfg: configAE):
        assert cfg.n_features, "n_features must be set in configAE"
        super().__init__()
        self.device = cfg.device
        self.model_dir = cfg.model_dir
        self.model_tag = f"{cfg.model_name}-{cfg.timestamp}"

        self.blocks(cfg)

    def blocks(self, cfg: configAE):
        """
        dynamically build symmetrical encoder/decoder blocks based on reference hidden layer dimension;
        note: `range(...)` creates a sequence of exponents from `n_layers-1` down to 0.
        """
        if cfg.exp_scaling:  # use exponents
            hidden_dims = [cfg.hidden_dim * (2**i) for i in range(cfg.n_layers - 1, -1, -1)]
        else:  # use multiples
            hidden_dims = [cfg.hidden_dim * (i + 1) for i in range(cfg.n_layers - 1, -1, -1)]

        # --- encoder ---
        ll_encoder = []
        in_features = cfg.n_features
        for h_dim in hidden_dims:
            ll_encoder.append(nn.Linear(in_features, h_dim))
            ll_encoder.append(nn.ReLU())
            ll_encoder.append(nn.LayerNorm(h_dim))
            ll_encoder.append(nn.Dropout(cfg.dropout_pct))
            in_features = h_dim
        ll_encoder.append(nn.Linear(in_features, cfg.embed_dim))
        self.encoder = nn.Sequential(*ll_encoder)

        # --- decoder ---
        ll_decoder = []
        in_features = cfg.embed_dim
        for h_dim in reversed(hidden_dims):
            ll_decoder.append(nn.Linear(in_features, h_dim))
            ll_decoder.append(nn.ReLU())
            ll_decoder.append(nn.LayerNorm(h_dim))
            ll_decoder.append(nn.Dropout(cfg.dropout_pct))
            in_features = h_dim
        ll_decoder.append(nn.Linear(in_features, cfg.n_features))
        self.decoder = nn.Sequential(*ll_decoder)

        # --- noise augmentation ---
        if cfg.noise_augment:
            self.augment = augFeatureNoise(p_feature=cfg.na_probability, sigma=cfg.na_sigma)
            print("feature noise augmentation layer added to autoencoder")
        else:
            self.augment = nn.Identity()

        # --- init weights ---
        self._tie_weights_() if cfg.tie_weights else None
        self.apply(self._init_weights_)
        self.to(cfg.device)

        n_params = sum(p.numel() for p in self.parameters()) / 1e3
        print(f"{self.__class__.__name__} initialised with {n_params:.2f}K parameters")

    def forward(self, x, encode=False):
        """can return either encoded representation or full reconstruction"""
        x = self.augment(x)
        encoded = self.encoder(x)
        if encode:
            return encoded
        reconstructed = self.decoder(encoded)
        return reconstructed, None  # return None for logits

    def save(self, verbose=False):
        fp = Path(self.model_dir) / f"{self.model_tag}/model.pth"
        fp.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        torch.save(self.state_dict(), fp)
        print(f"model saved to {fp}") if verbose else None

    def load(self, verbose=True):
        fp = Path(self.model_dir) / f"{self.model_tag}/model.pth"
        self.load_state_dict(torch.load(fp, map_location=self.device))
        print(f"model loaded from {fp}") if verbose else None

    def _init_weights_(self, module):
        """initialise weights of linear layers with kaiming uniform distribution"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _tie_weights_(self):
        """tie the decoder weights to the transpose of the encoder weights"""
        encoder_linears = [m for m in self.encoder if isinstance(m, nn.Linear)]
        decoder_linears = [m for m in self.decoder if isinstance(m, nn.Linear)]

        for l_encoder, l_decoder in zip(encoder_linears, reversed(decoder_linears)):
            l_decoder.weight = nn.Parameter(l_encoder.weight.t())


class L2NormLayer(nn.Module):
    """l2 normalisation layer that normalises each sample to unit length"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps  # small value to prevent division by zero

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        x = x / norms
        return x


class hybridAutoEncoder(baseAutoEncoder):
    """hybrid autoencoder with supervised classification head on top of the encoded representation"""

    def __init__(self, cfg: configAE):
        super().__init__(cfg)

        # --- classification head ---
        self.classifier = nn.Sequential(
            L2NormLayer(),
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_pct),
            nn.Linear(cfg.hidden_dim, cfg.ns_classes),
        )

        # --- init weights ---
        self.classifier.apply(self._init_weights_)
        self.to(cfg.device)

        n_params = sum(p.numel() for p in self.classifier.parameters()) / 1e3
        print(f"supervised classification head added with {n_params:.2f}K additional parameters")

    def forward(self, x, encode=False):
        """can return encoded representation or tuple of (full reconstruction, classification logits)"""
        x = self.augment(x)
        encoded = self.encoder(x)
        if encode:
            return encoded
        logits = self.classifier(encoded)
        reconstructed = self.decoder(encoded)
        return reconstructed, logits


def hybridLoss(reconstruction, xs, logits, ys, cfg: configAE):
    """
    compute hybrid loss for hybrid autoencoder;
    combines reconstruction and binary classification losses with weighting factor lambda;
    the classification loss also takes a weighting factor alpha to handle class imbalance.
    """
    # reconstruction loss
    mse_loss = nn.MSELoss()
    reconstruction_loss = mse_loss(reconstruction, xs)
    if not cfg.supervised:
        return reconstruction_loss

    # classification loss
    assert logits is not None, "logits must be provided for supervised hybrid autoencoder"
    targets = ys[:, 1].float()  # assuming ys is (grid ids, binary targets)
    alpha = min(cfg.wf_alpha, 50.0)  # cap alpha to prevent extreme weighting
    pos_weight = torch.tensor([alpha], device=cfg.device)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    classification_loss = bce_loss(logits.squeeze(), targets)
    return reconstruction_loss + cfg.wf_lambda * classification_loss


class EarlyStopping:
    """
    callable class to check for early stopping criterion; if criterion is met, load the best model weights saved so far.
    """

    def __init__(self, m, cfg: configAE):
        self.m = m
        self.save = cfg.save_model
        self.patience = cfg.es_patience

        # init tracking variables
        self.best_loss = float("inf")
        self.epochs_no_improve = 0

    def __call__(self, current_loss: float) -> bool:
        """
        check if early stopping criterion is met based on validation losses;
        return `True` if training should stop, `False` otherwise.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_no_improve = 0  # reset no improvement counter
            self.m.save() if self.save else None  # save the best model weights so far (?)
            return False
        else:
            self.epochs_no_improve += 1  # increment no improvement counter
            if self.epochs_no_improve >= self.patience:
                print(f"early stopping criterion met; best validation loss: {self.best_loss:.4f}")
                self.m.load() if self.save else None  # load the best model weights saved so far (?)
                return True
        return False


def trainAE(dls: tuple, model: baseAutoEncoder, cfg: configAE):
    """
    training loop for autoencoder with reconstruction (and classification if hybrid) loss;
     assumes `ys` has shape (batch_size, 2) with (grid ids, binary targets).
    """
    train_dls, valid_dls = dls
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    log_metrics = []

    if cfg.early_stopping:
        ES = EarlyStopping(model, cfg)

    with tqdm(range(cfg.n_epochs), desc="training") as pbar:
        for epoch in pbar:
            model.train()
            train_loss = 0.0
            for xs, ys in train_dls:
                optimizer.zero_grad()
                reconstruction, logits = model(xs)
                loss = hybridLoss(reconstruction, xs, logits, ys, cfg)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dls)

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for xs, ys in valid_dls:
                    reconstruction, logits = model(xs)
                    loss = hybridLoss(reconstruction, xs, logits, ys, cfg)
                    valid_loss += loss.item()
            valid_loss /= len(valid_dls)

            losses = {"train": train_loss, "valid": valid_loss}
            log_metrics.append(losses)
            pbar.set_postfix(losses)

            if cfg.early_stopping and ES(valid_loss):
                break

    return log_metrics


class featureExtractor:
    """
    extract features from trained autoencoder models with various configurations;
    centralises dataloader creation, feature extraction, and normalisation logic;
    returns encodings, supervised logits, or their combination as df with `id` as index.
    """

    def __init__(self, model, to, bs: int | None = None):
        self.model = model
        self.to = to
        self.bs = bs or configAE.batch_size
        self.device = model.device
        self.process()  # create dataloader and extract features

    def process(self):
        """
        extract both encodings and logits from model in single pass through data;
        returns self for method chaining.
        """
        # create dataloader
        ds = TensorDataset(
            torch.tensor(self.to.xs.values, dtype=torch.float32).to(self.device),
            torch.tensor(self.to.ys.values, dtype=torch.long).to(self.device),
        )
        dls = DataLoader(ds, batch_size=self.bs, shuffle=False)
        print(f"extracting encodings + logits with internal dataloader of batch size {self.bs}...")

        self.model.eval()
        encodings, logits, ids = [], [], []

        with torch.no_grad():
            for xs, ys in dls:
                b_encoded = self.model(xs, encode=True)
                b_logits = self.model.classifier(b_encoded)
                encodings.append(b_encoded.cpu().numpy())
                logits.append(b_logits.cpu().numpy())
                ids.append(ys[:, 0].cpu().numpy())

        lf = lambda x: np.concatenate(x, axis=0).squeeze()  # flatten lists of arrays into single arrays
        self.encodings = lf(encodings)
        self.logits = lf(logits)
        self.ids = lf(ids)

        print(f"{self.encodings.shape=} + {self.logits.shape=} > {self.ids.shape=}")
        return self

    def _normalise(self, features, zscore=True, l2norm=True):
        if zscore:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            print(f"z-score normalisation applied to encoded features")

        if l2norm:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(norms, 1e-9)  # avoid division by zero
            print(f"l2-normalisation applied to encoded samples")

        return features

    def _to_dataframe(self, features):
        df = pd.DataFrame(features, index=self.ids)
        df = df.sort_index().reset_index(names="id")
        assert sanityCheck(df), "!!!"
        return df

    def encodedFeatures(self, zscore=True, l2norm=True) -> pd.DataFrame:
        """
        get encodings from trained autoencoder;
        applies optional normalisation (z-score and/or l2-normalisation) to the encoded features.
        """
        features = self._normalise(self.encodings.copy(), zscore, l2norm)
        return self._to_dataframe(features)

    def supervisedLogits(self, sigmoid=True) -> pd.DataFrame:
        """
        get classification logits from trained hybrid autoencoder;
        optionally applies sigmoid to convert logits to probabilities for better interpretability.
        """
        logits = self.logits.copy()
        if sigmoid:
            logits = 1 / (1 + np.exp(-logits))  # sigmoid
            print("raw logits converted to probabilities using sigmoid")
        return self._to_dataframe(logits)

    def combinedFeatures(self, zscore=True, l2norm=True) -> pd.DataFrame:
        """
        get combined encodings and classification logits from trained hybrid autoencoder;
        applies optional normalisation (z-score and/or l2-normalisation) to the combined features.
        """
        logits = self.logits
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)  # ensure logits is 2d for concatenation
        combined = np.concatenate([self.encodings, logits], axis=1)
        features = self._normalise(combined, zscore, l2norm)
        return self._to_dataframe(features)


def getEncodedFeatures(model, to, zscore=True, l2norm=True, bs: int | None = None) -> pd.DataFrame:
    """
    get encoded features for entire dataset from trained autoencoder model and TabularPandas object;
    creates a dataloader internally to iterate through the data in batches using the provided batch size;
    applies optional normalisation (z-score and/or l2-normalisation) to the encoded features.

    returns a pandas DataFrame of encoded features with `id` as index;
    since `id` was passed as (int) target to TabularPandas, it is not affected by encoding/decoding.

    functionality is now centralised in `featureExtractor` class; kept for backward compatibility.
    """
    extractor = featureExtractor(model, to, bs=bs)
    return extractor.encodedFeatures(zscore=zscore, l2norm=l2norm)


class trainUtils:
    """collection of helper utils for model training, evaluation and optimisation"""

    @staticmethod
    def plotTrainingLosses(logs: list, cfg: configAE, save: str | None = None, verbose=False):
        """plot training and validation loss curves from training logs"""
        df = pd.DataFrame(logs).round(4)
        iters = int(df["valid"].idxmin())
        besties = df.loc[iters].to_dict()  # type:ignore
        cfg.metrics = ", ".join([f"{k}:{v}" for k, v in besties.items()])

        df.plot(kind="line", figsize=(10, 4), logy=True)
        plt.xlabel("epoch")
        plt.xlim(-0.5, len(logs) + 0.5)
        plt.ylabel("mse loss")
        plt.grid(which="both", linestyle=":")
        plt.title(f"@epoch {iters + 1}/{cfg.n_epochs}: {besties}")

        plt.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save, dpi=300, bbox_inches="tight")
            print(f"training loss curves saved to ./{save}") if verbose else None
            plt.close()
        plt.show()

    @staticmethod
    def getModelName(cfg: configAE) -> str:
        """auto-generate a model name based on configuration parameters"""
        name = "hae" if cfg.supervised else "bae"
        name += f"L{cfg.n_layers}"
        name += f"H{cfg.hidden_dim}" + ("e" if cfg.exp_scaling else "m")
        name += f"E{cfg.embed_dim}"
        return name

    @staticmethod
    def getPlotTitle(cfg) -> str:
        """auto-generate a plot title based on configuration parameters"""
        excludes = ["entity", "project", "model_dir", "n_epochs", "timestamp", "device", "save_model"]
        cfg = vars(cfg) if hasattr(cfg, "__dict__") else cfg
        defaults = {k: v for k, v in vars(configAE()).items()}
        overrides = {k: v for k, v in cfg.items() if (k not in excludes) and (v != defaults[k])}
        overrides = selfRNDR(overrides)  # shorten long values for readability
        title = " | ".join([f"{k}={v}" for k, v in overrides.items()])  # type:ignore
        return title

    @staticmethod
    def saveTrainingConfig(cfg, fp: str | Path, verbose=False):
        cfg = vars(cfg) if hasattr(cfg, "__dict__") else cfg
        with open(fp, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"training configuration saved to ./{fp}") if verbose else None

    @staticmethod
    def saveTrainingLog(log: List[dict], fp: str | Path, verbose=False):
        with open(fp, "w") as f:
            json.dump(log, f, indent=4)
        print(f"training log saved to ./{fp}") if verbose else None

    @staticmethod
    def getRequiredArgs(obj) -> list[str]:
        """return names of required arguments for a dataclass"""
        return [f.name for f in fields(obj) if f.default is MISSING and f.default_factory is MISSING]
