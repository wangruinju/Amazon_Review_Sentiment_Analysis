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
    category = fields.SelectField('Category', choices=[('pet', 'Pet'),
                                                       ('automotive', 'Automotive'),
                                                       ('music', 'Music')])
    review = fields.TextAreaField('Review:', validators=[Required()])

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
        category = submitted_data['category']

        # Retrieve values from form
        review = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])

        if category == 'pet':
            review = pet_vec.transform([review])
            my_prediction = pet_estimator.predict(review.multiply(pet_r))
            my_proba = pet_estimator.predict_proba(review.multiply(pet_r))

        if category == 'automotive':
            review = automotive_vec.transform([review])
            my_prediction = automotive_estimator.predict(review.multiply(automotive_r))
            my_proba = automotive_estimator.predict_proba(review.multiply(automotive_r))

        if category == 'music':
            review = music_vec.transform([review])
            my_prediction = music_estimator.predict(review.multiply(music_r))
            my_proba = music_estimator.predict_proba(review.multiply(music_r))

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
