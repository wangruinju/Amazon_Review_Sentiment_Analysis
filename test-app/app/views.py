import logging
import json
import numpy as np
import re
import string
from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required

from . import app, estimator, vec, target_names, r

logger = logging.getLogger('app')

class PredictForm(Form):
    """Fields for Predict"""
    review = fields.TextField('review:', validators=[Required()])

    submit = fields.SubmitField('Submit')

@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    predicted = None
    my_proba = None
    proba = None

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data

        # Retrieve values from form
        review = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])
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
