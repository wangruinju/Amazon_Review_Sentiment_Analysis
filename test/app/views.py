import logging
import json
import numpy as np
import re
import string
from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
from wtforms.widgets import TextArea

from . import app, target_names, automotive_estimator, music_estimator, pet_estimator,\
automotive_vec, music_vec, pet_vec, automotive_r, music_r, pet_r

logger = logging.getLogger('app')

class PredictForm(Form):
    """Fields for Predict"""
    category = fields.SelectField('Category', choices=[ ('Automotive', 'Automotive'),
                                                        ('Baby', 'Baby'),
                                                        ('Clothing Shoes and Jewelry', 'Clothing_Shoes_and_Jewelry'),	
                                                        ('Digital Music', 'Digital_Music'),	
                                                        ('Electronics', 'Electronics'),	
                                                        ('Grocery and Gourmet', 'Grocery_and_Gourmet'),	
                                                        ('Home and Kitchen', 'Home_and_Kitchen'),	
                                                        ('Kindle store', 'Kindle_Store'), 	
                                                        ('Pet Supplies', 'Pet_Supplies'),	
                                                        ('Sports and Outdoors', 'Sports_and_Outdoors'),	
                                                        ('Toys and Games', 'Toys_and_Games'),	
                                                        ('Video Games', 'Video_Games') ])
    review = fields.TextAreaField('Review:', validators=[Required()])

    submit = fields.SubmitField('Submit')

@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    target_names = ['Negative', 'Positive']
    predicted = None
    my_proba = None
    proba = None

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data
        category = submitted_data['category']
        category_names = ['Automotive', 'Baby', 'Clothing_Shoes_and_Jewelry',
                      'Digital_Music', 'Electronics', 'Grocery_and_Gourmet',
                      'Home_and_Kitchen', 'Kindle_Store', 'Pet_Supplies',
                      'Sports_and_Outdoors', 'Toys_and_Games', 'Video_Games']

        # Retrieve values from form
        review = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])
        for name in category_names:
            if category == name:
                model_loc = 'models/' + name + "_model.pkl"
                vec_loc = 'models/' + name + "_vector.pkl"
                r_loc = 'models/' + name + "_r.npz"
                # unpickle my model
                estimator = joblib.load(model_loc)
                vec = joblib.load(vec_loc)
                r = sparse.load_npz(r_loc)
                break

        review = vec.transform([review])
        my_prediction = estimator.predict(review.multiply(r))
        my_proba = estimator.predict_proba(review.multiply(r))
            
        # Return only the Predicted iris species
        predicted = target_names[int(my_prediction)]
        if my_prediction < 0.5:
            proba = str(round(my_proba[0][0]*100, 2))
        else:
            proba = str(round(my_proba[0][1]*100, 2))

    return render_template('index.html',
        form=form,
        prediction=predicted,
        prob = proba)
