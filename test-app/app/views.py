import logging
import json
import numpy as np
import re
import string
from flask import render_template
from flask_wtf import Form
from wtforms import fields, RadioField
from wtforms.validators import Required

from . import app, estimator, vec, target_names, r

logger = logging.getLogger('app')

class PredictForm(Form):
    """Fields for Category"""
    category = RadioField('category', choices=[('music','music'),('pet','pet')], validators=[Required()])
    """Fields for Predict"""
    review = fields.TextField('review:', validators=[Required()])

    submit = fields.SubmitField('Submit')

@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    predicted = None
    my_proba = None

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data

        # Retrieve values from form
        review = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])
        review = vec.transform([review])
        my_prediction = estimator.predict(review.multiply(r))

        # Return only the Predicted iris species
        predicted = target_names[int(my_prediction)]
        #my_proba = list(estimator.predict_proba(review))
        #my_proba = str(round(my_proba[int(my_prediction)]*100,2))
        #my_proba = str(round(estimator.predict_proba(review)[my_prediction]*100, 2))

    return render_template('index.html',
        form=form,
        prediction=predicted,
        prob = my_proba)
