import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.externals import joblib
from scipy import sparse
import gzip, re, string

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def pr(x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    df = getDF('reviews_Digital_Music_5.json.gz')
    df['reviewText'].fillna("unknown", inplace=True)

    X = df.drop(['overall'], axis = 1)
    y = (df['overall'] > 3).astype(float)
    texts = list(X['reviewText'])
    treated_text = []
    for text in texts:
        treated_text.append(re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', text))
    
    vec = TfidfVectorizer(ngram_range=(1,2),
            min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
            smooth_idf=1, sublinear_tf=1 )
    x = vec.fit_transform(treated_text)
    y = y.values
    r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
    clf = LogisticRegression(C = 10, dual=True, penalty = "l2", n_jobs = -1)
    x_nb = x.multiply(r)
    clf.fit(x_nb, y)
    # clf.predict(x_test.multiply(r))
    joblib.dump(clf, 'models/review_model.pkl')
    joblib.dump(vec, 'models/review_vector.pkl')
    sparse.save_npz('models/review_r.npz', r)
