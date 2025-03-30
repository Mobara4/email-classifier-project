# converting text data into numerical vectors for preprocessing.

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def vectorize_text(df, text_cols):
    tfidf = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    vectors = [tfidf.fit_transform(df[col].astype(str)).toarray() for col in text_cols]
    return np.concatenate(vectors, axis=1)
