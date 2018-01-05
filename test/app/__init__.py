import logging
from sklearn.externals import joblib

from flask import Flask
from scipy import sparse

# create logger for app
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)


#tokenize = tokenize()
app = Flask(__name__)
app.config.from_object("app.config")


#app.add_template_global(tokenize, 'tokenize')

# unpickle my model
automotive_estimator = joblib.load('models/Automotive_model.pkl')
automotive_vec = joblib.load('models/Automotive_vector.pkl')
target_names = ['Negative', 'Positive']
automotive_r = sparse.load_npz('models/Automotive_r.npz')

music_estimator = joblib.load('models/Music_model.pkl')
music_vec = joblib.load('models/Music_vector.pkl')
music_r = sparse.load_npz('models/Music_r.npz')

pet_estimator = joblib.load('models/Pet_model.pkl')
pet_vec = joblib.load('models/Pet_vector.pkl')
pet_r = sparse.load_npz('models/Pet_r.npz')


from .views import *   # flake8: noqa

# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    """Page Not Found"""
    return render_template('404.html'), 404
