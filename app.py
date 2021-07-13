#!/usr/bin/env python
# coding: utf-8

# In[8]:


from flask import Flask, render_template, session, redirect, url_for, session, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired
import numpy as np
from tensorflow.keras.models import load_model
import joblib


# In[3]:


def return_prediction(model, scaler, onehot, lab_enc, sample_json):
    cr_score = sample_json['CreditScore']
    geo = sample_json['Geography']
    gen = sample_json['Gender']
    age = sample_json['Age']
    ten = sample_json['Tenure']
    bal = sample_json['Balance']
    num = sample_json['NumOfProducts']
    has_card = sample_json['HasCrCard']
    is_active = sample_json['IsActiveMember']
    sal = sample_json['EstimatedSalary']
    
    pred=[[cr_score,geo,gen,age,ten,bal,num,has_card,is_active,sal]]
    pred[0][2] = lab_enc.transform([pred[0][2]])[0]
    pred = np.array(onehot.transform(pred))
    return (model.predict(scaler.transform(pred)))[0][0]*100


# In[18]:


app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'someRandomKey'

# Loading the model and scaler
model = load_model("customer_retention_1.h5")
scaler = joblib.load("ann_scaler.pkl")
ohencoder = joblib.load("onehot_col_transformer.pkl")
labencoder = joblib.load("label_enc.pkl")

# Now create a WTForm Class
class FlowerForm(FlaskForm):
    cr_score = IntegerField('Credit Score', validators=[DataRequired()])
    geo = StringField('Geography', validators=[DataRequired()])
    gen = StringField('Gender', validators=[DataRequired()])
    age = IntegerField('Age', validators=[DataRequired()])
    ten = IntegerField('Tenure', validators=[DataRequired()])
    bal = IntegerField('Balance', validators=[DataRequired()])
    num = IntegerField('Number Of Products', validators=[DataRequired()])
    has_card = IntegerField('Has Credit Card', validators=[DataRequired()])
    is_active = IntegerField('Is Active Member', validators=[DataRequired()])
    sal = IntegerField('Estimated Salary', validators=[DataRequired()])
    
    submit = SubmitField('Analyze')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = FlowerForm(request.form)
    # If the form is valid on submission
    if request.method == "POST" and form.validate():
        # Grab the data from the input on the form.
        session["cr_score"] = form.cr_score.data
        session["geo"] = form.geo.data
        session["gen"] = form.gen.data
        session["age"] = form.age.data
        session["ten"] = form.ten.data
        session["bal"] = form.bal.data
        session["num"] = form.num.data
        session["has_card"] = form.has_card.data
        session["is_active"] = form.is_active.data
        session["sal"] = form.sal.data
        
        return redirect(url_for("prediction"))
    print(form.errors)
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    #Defining content dictionary
    content = {}
    
    content["CreditScore"] = float(session["cr_score"])
    content["Geography"] = str(session["geo"])
    content["Gender"] = str(session["gen"])
    content["Age"] = float(session["age"])
    content["Tenure"] = float(session["ten"])
    content["Balance"] = float(session["bal"])
    content["NumOfProducts"] = float(session["num"])
    content["HasCrCard"] = float(session["has_card"])
    content["IsActiveMember"] = float(session["is_active"])
    content["EstimatedSalary"] = float(session["sal"])


    results = return_prediction(model = model, scaler = scaler, onehot = ohencoder, lab_enc=labencoder, sample_json = content)
    return render_template('prediction.html',results=results)
        
if __name__ == "__main__":
    app.run(port=5000, debug=True)


# In[ ]:




