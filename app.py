
# coding: utf-8

#from crypt import methods
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
import os

app = Flask("__name__",static_url_path='/static')

df_1=pd.read_csv("first_telc.csv")

q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")

@app.route("/home")
def home():
	return render_template('home.html', query="")

@app.route("/upload_csv",methods=['GET','POST'])
def upload_csv():
    if request.method =='POST':
        file = request.files['csvfile']
        filepath = os.path.join('static',file.filename)
        file.save(filepath)
        telco_base_data = pd.read_csv(filepath)
        telco_data = telco_base_data.copy()
        telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
        telco_data.dropna(how = 'any', inplace = True)
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)
        telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
        telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)
        telco_data_dummies = pd.get_dummies(telco_data)
        plt.figure(figsize=(20,8))
        telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
        # fig = plt.figure(figsize = (15,20))
        # ax = fig.gca()
        # telco_data.hist(ax = ax)
        # plt.figure(figsize=(20,8))
        # telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
        #plt.figure(figsize=(12,12))
        #sns.heatmap(telco_data_dummies.corr(), cmap="Paired")
        # plt.plot(df[variable])
        # X = df.iloc[:,0:20]  #independent columns
        # y = df.iloc[:,-1]    #target column i.e price range

        # # apply SelectKBest class to extract top 10 best features
        # bestfeatures = SelectKBest(score_func=chi2, k=10)
        # fit = bestfeatures.fit(X,y)
        # dfscores = pd.DataFrame(fit.scores_)
        # dfcolumns = pd.DataFrame(X.columns)

        # #concat two dataframes for better visualization 
        # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        # featureScores.columns = ['Specs','Score']
        # plt.figure(figsize=(20,5))
        # sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")
        # plt.box(False)
        # plt.title('Feature importance', fontsize=16)
        # plt.xlabel('\n Features', fontsize=14)
        # plt.ylabel('Importance \n', fontsize=14)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        #plt.show()
        imagepath = os.path.join('static','feature'+'.png')
        plt.savefig(imagepath)
        return render_template('feature.html', image = imagepath) 

    return render_template('upload_csv.html')

# @app.route('/dash',methods = ['GET' , 'POST'])
# def dash():
#     if request.method == 'POST':
#         variable = request.form['variable']
#         df = pd.read_csv('static/test.csv')
#         # plt.plot(df[variable])
#         X = df.iloc[:,0:20]  #independent columns
#         y = df.iloc[:,-1]    #target column i.e price range

#         # apply SelectKBest class to extract top 10 best features
#         bestfeatures = SelectKBest(score_func=chi2, k=10)
#         fit = bestfeatures.fit(X,y)
#         dfscores = pd.DataFrame(fit.scores_)
#         dfcolumns = pd.DataFrame(X.columns)

#         #concat two dataframes for better visualization 
#         featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#         featureScores.columns = ['Specs','Score']
#         plt.figure(figsize=(20,5))
#         sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")
#         plt.box(False)
#         plt.title('Feature importance', fontsize=16)
#         plt.xlabel('\n Features', fontsize=14)
#         plt.ylabel('Importance \n', fontsize=14)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         #plt.show()
#         imagepath = os.path.join('static','feature'+'.png')
#         plt.savefig(imagepath)
#         return render_template('feature.html',image=imagepath)

@app.route("/home", methods=['POST'])
def predict():
    
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''
    

    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']

    model = pickle.load(open("model.sav", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    
    new_df = pd.DataFrame(data, columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    #drop column customerID and tenure
    df_2.drop(columns= ['tenure'], axis=1, inplace=True)   
    
    
    
    
    new_df__dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']])
    
    
    #final_df=pd.concat([new_df__dummies, new_dummy], axis=1)
        
    
    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13'], 
                           query14 = request.form['query14'], 
                           query15 = request.form['query15'], 
                           query16 = request.form['query16'], 
                           query17 = request.form['query17'],
                           query18 = request.form['query18'], 
                           query19 = request.form['query19'])
    

if(__name__=="__main__"):
    app.run(debug=True)
        
        




