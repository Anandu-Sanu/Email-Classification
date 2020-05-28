# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:43:40 2020

@author: Anandhu Sanu
"""
from flask import Flask, request,render_template
from flask_cors import CORS
import os
from sklearn.externals import joblib
import flask

app = Flask(__name__)
CORS(app)

app=flask.Flask(__name__,template_folder='F:\\ExcelR\\project\\templates')


#NBmodel = open('NBmodel.pkl','rb')
#vectorizer=open('vectorizer.pkl','rb')
clf = joblib.load('NBmodel.pkl')
cv=joblib.load('Tfidf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    temp=request.get_data(as_text=True)
    new=[]
    new.append(temp)
    message=cv.transform(new)
    pred = clf.predict(message)
    
    if pred[0]=='Abusive' :
        return render_template('index.html',prediction = 'Message Classified As Abusive Message.')
    else:
        return render_template('index.html',prediction = 'Message Classified As Non Abusive.')
    

    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
