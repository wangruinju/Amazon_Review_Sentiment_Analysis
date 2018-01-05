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

    filenames = ['reviews_Kindle_Store_5.json.gz',
             'reviews_Grocery_and_Gourmet_Food_5.json.gz',
             'reviews_Automotive_5.json.gz',
             'reviews_Digital_Music_5.json.gz',
             'reviews_Pet_Supplies_5.json.gz',
             'reviews_Baby_5.json.gz',
             'reviews_Sports_and_Outdoors_5.json.gz',
             'reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
             'reviews_Toys_and_Games_5.json.gz',
             'reviews_Home_and_Kitchen_5.json.gz',
             'reviews_Video_Games_5.json.gz',
             'reviews_Electronics_5.json.gz'
            ]
    for name in filenames:
        df = getDF(name)
        df['reviewText'].fillna("unknown", inplace=True)

        treated_text = []
        for index, row in df.iterrows():
            treated_text.append(re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', row['reviewText']))
        
        vec = TfidfVectorizer(ngram_range=(1,2),
                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                smooth_idf=1, sublinear_tf=1 )
        x = vec.fit_transform(treated_text)
        y = (df['overall'] > 3).astype(float).values
        r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        clf = LogisticRegression(C = 10, dual=True, penalty = "l2", n_jobs = -1)
        x_nb = x.multiply(r)
        clf.fit(x_nb, y)
        # clf.predict(x_test.multiply(r))
        model_loc = name + "_model.pkl"
        vec_loc = name + "_vector.pkl"
        r_loc = name + "_r.npz"
        joblib.dump(clf, model_loc)
        joblib.dump(vec, vec_loc)
        sparse.save_npz(r_loc, r)
