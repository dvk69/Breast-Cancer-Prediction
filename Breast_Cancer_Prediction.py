#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt #for plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix
from math import sqrt
from statistics import mean


# In[61]:


# Collecting the data


# In[62]:


data = pd.read_csv(r"C:\Users\gundu harsha\Downloads\data (1).csv",header=0)


# In[63]:


data.shape


# In[64]:


data.head()


# In[65]:


# Exploratory Data Analysis


# In[66]:


data.info()


# In[67]:


# Detecting the missing values
data.isna() 


# In[95]:


# Dropping the column
data = data.dropna(axis='columns') 


# In[83]:


data.describe() 


# In[86]:


M_B = data.diagnosis.unique() # Finding out the unique values for M and B
M_B


# In[87]:


data.diagnosis.value_counts() # Counting Unique Values


# In[15]:


# Data Visualization


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  
import seaborn as sns  
import plotly.express as px #high-level API for creating figures
import plotly.graph_objects as go


# In[89]:


plt.figure(figsize=(6, 2))
plt.subplot(2, 2, 2)
plt.hist(data.diagnosis)
plt.title("Diagnosis count")
plt.xlabel("Diagnosis")
plt.ylabel("Count");


# In[71]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis") # Column is variable and row is observation
plt.show()


# In[19]:


# Data Filtering


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


data.head()


# In[22]:


labelencoder_Y = LabelEncoder() #LabelEncoder can be used to normalize labels.
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis) 


# In[90]:


data.tail()


# In[75]:


print(data.diagnosis.value_counts())


# In[76]:


# Finally, we can observe categorical values changed to 0 and 1 in this output.


# In[77]:


#Find the correlation between other features, mean features only
cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[78]:


# plotting the Graph
plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 100, 20, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[28]:


#Model Implementation


# In[29]:


##Import Machine Learning Models and Train Test Splitting


# In[30]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[32]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[33]:


#verify the Model Accuracy, Errors and it's Validations


# In[34]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[35]:


data.columns


# In[36]:


#Take the dependent and independent feature for prediction
prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[79]:


X = data[prediction_feature]
print(X)


# In[82]:


y = data.diagnosis
print(y)


# In[39]:


#Split the dataset into TrainingSet and TestingSet by 45% and set the 20 fixed records


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=20)
X_train


# In[41]:


#Perform Feature Standerd Scalling


# In[42]:


# Scale the data to keep all the values in the same magnitude of 0 -1 

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[43]:


# ML Model Selecting and Model PredPrediction


# In[44]:


def model_building(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    
    return (score, accuracy, predictions)


# In[46]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}
print(models_list)


# In[47]:


print(list(models_list.keys()))
print(list(models_list.values()))

print(zip(list(models_list.keys()), list(models_list.values())))


# In[48]:


#Model Implementing


# In[49]:


df_prediction = []
confusion_matrixs = []
df_prediction_cols = [ 'model_name', 'score', 'accuracy_score' , "accuracy_percentage"]

for name, model in zip(list(models_list.keys()), list(models_list.values())):
    
    (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test )
    
    print("\n\nClassification Report of '"+ str(name), "'\n")
    
    print(classification_report(y_test, predictions))

    df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])
    
    # For Showing Metrics
    confusion_matrixs.append(confusion_matrix(y_test, predictions))
    
        
df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)


# In[91]:


len(confusion_matrixs)


# In[51]:


df_pred #While Predicting we can store model's score and prediction values to new generated dataframe


# In[94]:


#printing the hightest accuracy score using sort values
df_pred.sort_values('score', ascending=False)
df_pred.sort_values('accuracy_score', ascending=False)


# In[53]:


#K-Fold
len(data)


# In[54]:


cv_score = cross_validate(LogisticRegression(), X, y, cv=3,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)

pd.DataFrame(cv_score).describe().T


# In[55]:


#defining a function for cross validation scorring for multiple ML models


# In[56]:


def cross_val_scorring(model):

    
    model.fit(data[prediction_feature], data[targeted_feature])
 
    
    predictions = model.predict(data[prediction_feature])    
    accuracy = accuracy_score(predictions, data[targeted_feature])
    print("\nFull-Data Accuracy:", round(accuracy, 2))
    print("Cross Validation Score of'"+ str(name), "'\n")
    
    
    # Initialize K folds.
    kFold = KFold(n_splits=5) # defining 5 diffrent data folds
    
    err = []
    
    for train_index, test_index in kFold.split(data):

        # Data Spliting via fold indexes
        X_train = data[prediction_feature].iloc[train_index, :]
        y_train = data[targeted_feature].iloc[train_index] # all targeted features trains
        
        X_test = data[prediction_feature].iloc[test_index, :] # testing all rows and cols
        y_test = data[targeted_feature].iloc[test_index] # all targeted tests
        
        # Again Model Fitting
        model.fit(X_train, y_train)

        err.append(model.score(X_train, y_train))
        
        print("Score:", round(np.mean(err),  2) )
#Call the function to know the cross validation function by mean for our select model predictions.

for name, model in zip(list(models_list.keys()), list(models_list.values())):
    cross_val_scorring(model)


# In[85]:


#GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.

from  sklearn.model_selection import GridSearchCV 

model = RandomForestClassifier()


# Tunning Params
random_grid = {'bootstrap': [1, 0],
 'max_depth': [40, 50, None], # 10, 20, 30, 60, 70, 100,
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2], # , 4
 'min_samples_split': [2, 5], # , 10
 'n_estimators': [200, 400]} # , 600, 800, 1000, 1200, 1400, 1600, 1800, 2000

# Implement GridSearchCV
gsc = GridSearchCV(model, random_grid, cv=10) # 10 Cross Validation

# Model Fitting
gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# In[ ]:




