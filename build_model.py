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

    category_names = ['Automotive', 'Baby', 'Clothing_Shoes_and_Jewelry',
                      'Digital_Music', 'Electronics', 'Grocery_and_Gourmet_Food',
                      'Home_and_Kitchen', 'Kindle_Store', 'Pet_Supplies',
                      'Sports_and_Outdoors', 'Toys_and_Games', 'Video_Games']

    for name in category_names:
        fileloc = 'data/reviews_' + name + '_5.json.gz'
        df = getDF(fileloc)
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
        # pickle your model for each category
        model_loc = 'models/reviews_' + name + "_5.json.gz_model.pkl"
        vec_loc = 'models/reviews_' + name + "_5.json.gz_vector.pkl"
        r_loc = 'models/reviews_' + name + "_5.json.gz_r.npz"
        joblib.dump(clf, model_loc)
        joblib.dump(vec, vec_loc)
        sparse.save_npz(r_loc, r)
