def individual_accuracies(y_true, y_pred, ids):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'id': ids})
    # FutureWarning vermeiden mit include_groups=False
    acc_per = df.groupby('id').apply(
        lambda g: (g.y_true == g.y_pred).mean(),
        include_groups=False
    )
    fem_ids = df.loc[df.y_true == 0, 'id'].unique()
    mal_ids = df.loc[df.y_true == 1, 'id'].unique()
    fem_acc = acc_per.loc[fem_ids].mean()
    mal_acc = acc_per.loc[mal_ids].mean()
    bal = 0.5 * (fem_acc + mal_acc)
    return fem_acc, mal_acc, bal

def individual_majority_stats(y_true, y_pred, ids):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'id': ids})
    # auch hier include_groups=False
    pct_per = df.groupby('id').apply(
        lambda g: (g.y_true == g.y_pred).mean(),
        include_groups=False
    )
    total = pct_per.shape[0]
    correct = int((pct_per > 0.5).sum())
    wrong   = total - correct
    pct     = correct / total
    return correct, wrong, pct
