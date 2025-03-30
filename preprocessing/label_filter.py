# filtering the labels for the purpose of optimized classification.
def filter_labels(df, label_cols, min_samples):
    for col in label_cols:
        counts = df[col].value_counts()
        df = df[df[col].isin(counts[counts >= min_samples].index)]
    return df
