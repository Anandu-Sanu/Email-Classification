# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:49:29 2020

@author: Anandhu Sanu
"""
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import classification_report,confusion_matrix
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split #importing train_test_split function
from sklearn.feature_extraction.text import TfidfVectorizer

email = pd.read_csv("F:\\ExcelR\\project\\emails.csv", index_col=0)

#studying/cleaning the data and performing EDA
emailDataDups=email.drop_duplicates(subset ="content",keep="first") # droping the rows with duplicate messages.

dupas=emailDataDups['content'].duplicated() #checking if duplicated contents are deleted.
print(dupas.any()) #prints if any duplicate rows are present

emailData = pd.DataFrame(emailDataDups)


print(emailData)
emailData.head()
emailData.dtypes
emailData.describe()
empty=emailData.isna() #checking for null variables
print(empty.any())
emailData.nunique() #gives the unique item count
dups=emailData.duplicated() #checking for duplicate rows
print(dups.any()) #prints if any duplicate rows are present

emailData['Length']=emailData['content'].apply(len) #finding the lengh of the Message
emailData.head()

#EDA
plt.rc("font", size=15)
emailData.Class.value_counts(sort=True).plot(kind='bar') #visualizing the data
emailData['Length'].plot(bins=100,kind='hist') #freequency dist for the length of the msgs.
emailData.Length.describe()
emailData.hist(column='Length',by='Class',bins=50)#from this we came to know that the abusive messages are longer than the non abusive ones.

#preprocessig the data.
#viewing the sportwords imported from nltk
emailData['content']
emailData['content'] = emailData['content'].map(lambda x: re.sub(r'\W+', ' ', x)) #removing all the regular expressions from the message.
#now removing the stopwords
stopwrd=stopwords.words('english')
emailData['content'] = emailData['content'].map(lambda x:  ' '.join([word for word in x.split() if word not in (stopwrd)])) #removing all the stop words from the message.
#building model
x=emailData.content
y=emailData.Class
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=0)

#checking the shape
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

vectorizer= TfidfVectorizer()
tfidf = vectorizer.fit(x_train)
xtest = vectorizer.transform(x_test)
xtrain = vectorizer.transform(x_train)

print(xtrain.shape)
print(xtest.shape)
print(y_train.shape)
print(y_test.shape)

# import SMOTE module from imblearn library for balancing the data. 

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
xtrain, ytrain = sm.fit_sample(xtrain, y_train)


# Applying k-Fold Cross Validation

from sklearn.model_selection import KFold, cross_val_score

kfold= KFold(n_splits=5, shuffle=False, random_state=None)

#multinomial NB DIRECTLY
#creating a multinomial model of naive bayes
MultiNomialmodel=MultinomialNB()
MultiNomialmodel.fit(xtrain,ytrain)
predictions = MultiNomialmodel.predict(xtest)
print ("Cross_Val_score : ",cross_val_score(MultiNomialmodel, X=xtrain, y=ytrain, cv=kfold, n_jobs=1,scoring='accuracy').mean())

print("Classification Report",classification_report(y_test,predictions))
confusionMAtrix=confusion_matrix(y_test,predictions)
pd.DataFrame(data=confusionMAtrix, columns = ['positive','negative'], index=['positive','negative'])
    

from sklearn.externals import joblib
joblib.dump(MultiNomialmodel, 'NBmodel.pkl')
joblib.dump(tfidf, 'Tfidf.pkl')
