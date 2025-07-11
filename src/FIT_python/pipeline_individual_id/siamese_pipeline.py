import numpy as np
import pandas as pd
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression

from .distance_metrics import compute_distances


class TripletDataset(Dataset):
    """Randomly generate triplets from a dataframe."""

    def __init__(
        self, df: pd.DataFrame, feature_cols: List[str], id_col: str = "individual_id"
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.features = feature_cols
        self.id_col = id_col
        self.by_id: Dict[str, List[int]] = {}
        for idx, ind in enumerate(self.df[id_col]):
            self.by_id.setdefault(str(ind), []).append(idx)
        self.ids = list(self.by_id.keys())

    def __len__(self) -> int:  # number of anchors
        return len(self.df)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng()
        anchor = self.df.loc[idx, self.features].values.astype(np.float32)
        anchor_id = str(self.df.loc[idx, self.id_col])

        pos_choices = [i for i in self.by_id[anchor_id] if i != idx]
        if not pos_choices:
            pos_choices = [idx]
        pos_idx = rng.choice(pos_choices)
        positive = self.df.loc[pos_idx, self.features].values.astype(np.float32)

        neg_id = rng.choice([i for i in self.ids if i != anchor_id])
        neg_idx = rng.choice(self.by_id[neg_id])
        negative = self.df.loc[neg_idx, self.features].values.astype(np.float32)
        return anchor, positive, negative


def build_dataloader(
    df: pd.DataFrame, feature_cols: List[str], batch_size: int
) -> DataLoader:
    ds = TripletDataset(df, feature_cols)
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
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> SiameseNet:
    dataloader = build_dataloader(df, feature_cols, batch_size)
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


def _pairwise_dataset(embeddings: np.ndarray, ids: List[str], n_samples: int = 10000):
    rng = np.random.default_rng(0)
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
    sex_features: Optional[List[str]] = None,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> List[Dict]:
    """Train a siamese network and evaluate on validation comparisons."""

    use_cols = feature_cols + (sex_features or [])
    df_train = train_df.copy()
    df_train[use_cols] = df_train[use_cols].apply(pd.to_numeric, errors="coerce")
    df_train = df_train.dropna(subset=use_cols)

    net = train_siamese(
        df_train,
        use_cols,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    embeddings = net.transform(df_train[use_cols])
    ids = df_train["individual_id"].astype(str).tolist()
    X_cls, y_cls = _pairwise_dataset(embeddings, ids)
    clf = LogisticRegression(max_iter=200).fit(X_cls, y_cls)

    results: List[Dict] = []
    for comp in val_comparisons:
        idx_a = comp["samples_a"]
        idx_b = comp["samples_b"]
        emb_a = embeddings[idx_a].mean(axis=0)
        emb_b = embeddings[idx_b].mean(axis=0)
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
