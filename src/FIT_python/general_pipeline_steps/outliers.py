"""Preconfigured outlier cleaning transformers."""

from .outlier_wrapper import OutlierCleanerTransformer

OUTLIERS = {
    "clip_90": OutlierCleanerTransformer(method="clip", lower_quantile=0.05, upper_quantile=0.95),
    "clip_98": OutlierCleanerTransformer(method="clip", lower_quantile=0.01, upper_quantile=0.99),
    "zscore_3": OutlierCleanerTransformer(method="zscore", z_thresh=3.0),
}

# --- 1) Centroid‑basierte Outlier‑Markierung pro Gruppe ---
def mark_outliers_centroid(df, bandwidth=0.5, percentile=1):
    """
    Berechnet für jede individual_id separat:
      1. Den Centroid der UMAP-Punkte.
      2. Eine Gauß‑KDE um diesen Centroid (Varianz = bandwidth^2).
      3. Die Dichte jedes echten Punktes unter diesem Kernel.
      4. Markiert als Outlier alle Punkte unterhalb des gegebenen Perzentils.
    """
    df = df.copy()
    df['centroid_density']    = np.nan
    df['is_outlier_centroid'] = False
    
    for ind, idx in df.groupby('individual_id').groups.items():
        pts = df.loc[idx, ['UMAP1','UMAP2']].values
        # 1) Centroid
        cx, cy = pts.mean(axis=0)
        # 2) Kernel um Centroid
        cov    = [[bandwidth**2, 0], [0, bandwidth**2]]
        kernel = multivariate_normal(mean=[cx, cy], cov=cov)
        # 3) Dichte-Bewertung
        dens   = kernel.pdf(pts)
        # 4) Schwellenwert am Perzentil
        thresh = np.percentile(dens, percentile)
        
        df.loc[idx, 'centroid_density']    = dens
        df.loc[idx, 'is_outlier_centroid'] = dens < thresh
        
    return df