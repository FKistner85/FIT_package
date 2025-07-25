import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression

from FIT_python.config import GLOBAL_RANDOM_SEED

from .distance_metrics import compute_distances
from .utils import map_indices


class TripletDataset(Dataset):
    """Randomly generate triplets from a dataframe.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal random number generator. Defaults to
        ``GLOBAL_RANDOM_SEED``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "individual_id",
        sample_col: str = "id",
        *,
        seed: int = GLOBAL_RANDOM_SEED,
    ) -> None:
        df = df.copy()
        df[sample_col] = df[sample_col].astype(str)
        self.df = df.set_index(sample_col)
        self.row_ids = list(self.df.index)
        self.features = feature_cols
        self.id_col = id_col
        self.by_id: Dict[str, List[str]] = {}
        for rid, ind in zip(self.df.index, self.df[id_col]):
            self.by_id.setdefault(str(ind), []).append(rid)
        self.ids = list(self.by_id.keys())
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:  # number of anchors
        return len(self.df)

    def __getitem__(self, idx: int):
        row_id = self.row_ids[idx]
        anchor = self.df.loc[row_id, self.features].to_numpy(dtype=np.float32)
        anchor_id = str(self.df.loc[row_id, self.id_col])

        pos_choices = [i for i in self.by_id[anchor_id] if i != row_id]
        if not pos_choices:
            pos_choices = [row_id]
        pos_id = self.rng.choice(pos_choices)
        positive = self.df.loc[pos_id, self.features].to_numpy(dtype=np.float32)

        neg_ind = self.rng.choice([i for i in self.ids if i != anchor_id])
        neg_id = self.rng.choice(self.by_id[neg_ind])
        negative = self.df.loc[neg_id, self.features].to_numpy(dtype=np.float32)
        return anchor, positive, negative


def build_dataloader(
    df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int,
    *,
    sample_col: str = "id",
    seed: int = GLOBAL_RANDOM_SEED,
) -> DataLoader:
    ds = TripletDataset(df, feature_cols, sample_col=sample_col, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


class SiameseNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = (
            X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        )
        with torch.no_grad():
            t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            emb = self.model(t)
        return emb.cpu().numpy()


def train_siamese(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    sample_col: str = "id",
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = GLOBAL_RANDOM_SEED,
) -> SiameseNet:
    dataloader = build_dataloader(
        df, feature_cols, batch_size, sample_col=sample_col, seed=seed
    )
    net = SiameseNet(len(feature_cols), embedding_dim, hidden_dim)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for _ in range(epochs):
        for anc, pos, neg in dataloader:
            anc = anc.to(net.device)
            pos = pos.to(net.device)
            neg = neg.to(net.device)
            optim.zero_grad()
            out_a = net(anc)
            out_p = net(pos)
            out_n = net(neg)
            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optim.step()
    return net


def _pairwise_dataset(
    embeddings: np.ndarray,
    ids: List[str],
    n_samples: int = 10000,
    *,
    seed: int = GLOBAL_RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return pairwise distance features and labels."""
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    idx1 = rng.integers(0, n, size=n_samples)
    idx2 = rng.integers(0, n, size=n_samples)
    X_feats = []
    y = []
    for i1, i2 in zip(idx1, idx2):
        dists = compute_distances(embeddings[i1], embeddings[i2])
        X_feats.append(list(dists.values()))
        y.append(int(ids[i1] == ids[i2]))
    return np.array(X_feats), np.array(y)


def run(
    train_df: pd.DataFrame,
    val_comparisons: List[Dict],
    feature_cols: List[str],
    *,
    sample_col: str = "id",
    sex_features: Optional[List[str]] = None,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_df: pd.DataFrame | None = None,
    seed: int = GLOBAL_RANDOM_SEED,
) -> List[Dict]:
    """Train a siamese network and evaluate on validation comparisons.

    Parameters
    ----------
    seed : int, optional
        Seed for random sampling in the training process. Defaults to
        ``GLOBAL_RANDOM_SEED``.
    """

    use_cols = feature_cols + (sex_features or [])
    df_train = train_df.copy()
    df_train[use_cols] = df_train[use_cols].apply(pd.to_numeric, errors="coerce")
    df_train = df_train.dropna(subset=use_cols)

    net = train_siamese(
        df_train,
        use_cols,
        sample_col=sample_col,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )

    embeddings_arr = net.transform(df_train[use_cols])
    df_train[sample_col] = df_train[sample_col].astype(str)
    emb_df = pd.DataFrame(embeddings_arr, index=df_train[sample_col])

    if val_df is not None:
        df_val = val_df.copy()
        df_val[use_cols] = df_val[use_cols].apply(pd.to_numeric, errors="coerce")
        df_val = df_val.dropna(subset=use_cols)
        df_val[sample_col] = df_val[sample_col].astype(str)
        emb_val = net.transform(df_val[use_cols])
        emb_val_df = pd.DataFrame(emb_val, index=df_val[sample_col])
        emb_df = pd.concat([emb_df, emb_val_df])

    idx_to_id = emb_df.index.to_series().reset_index(drop=True)

    ids = df_train["individual_id"].astype(str).tolist()
    X_cls, y_cls = _pairwise_dataset(embeddings_arr, ids, seed=seed)
    clf = LogisticRegression(max_iter=200).fit(X_cls, y_cls)

    results: List[Dict] = []
    for comp in val_comparisons:
        ids_a = map_indices(comp["samples_a"], idx_to_id, emb_df.index)
        ids_b = map_indices(comp["samples_b"], idx_to_id, emb_df.index)
        emb_a = emb_df.loc[ids_a].to_numpy().mean(axis=0)
        emb_b = emb_df.loc[ids_b].to_numpy().mean(axis=0)
        dists = compute_distances(emb_a, emb_b)
        X_feat = np.array([list(dists.values())])
        proba = clf.predict_proba(X_feat)[0, 1]
        results.append(
            {
                "trail_a_id": comp["trail_a_id"],
                "trail_b_id": comp["trail_b_id"],
                "pred_same": float(proba),
            }
        )
    return results
