# -*- coding: utf-8 -*-
import flask
from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

with open(f'models/model.pkl','rb') as f:
    model=pickle.load(f)
    
app=flask.Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET','POST'])

def main():
    if flask.request.method=="GET":
        return (flask.render_template("index.html"))
    
    if flask.request.method=="POST":
        ph=flask.request.form['ph']
        Hardness=flask.request.form['Hardness']
        Solids=flask.request.form['Solids']
        Chloramines=flask.request.form['Chloramines']
        Sulfate=flask.request.form['Sulfate']
        Conductivity=flask.request.form['Conductivity']
        Organic_carbon=flask.request.form['Organic_carbon']
        Trihalomethanes=flask.request.form['Trihalomethanes']
        Turbidity=flask.request.form['Turbidity']
        
        input_variables=pd.DataFrame([[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]],columns=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'],index=['input'])
        prediction=model.predict(input_variables)[0]
        return flask.render_template('index.html',original_input={'ph':ph,'Hardness':Hardness,'Solids':Solids,'Chloramines':Chloramines,'Sulfate':Sulfate,'Conductivity':Conductivity,'Organic_carbon':Organic_carbon,'Trihalomethanes':Trihalomethanes,'Turbidity':Turbidity},result=prediction,)


if __name__=='__main__':
            app.run()

