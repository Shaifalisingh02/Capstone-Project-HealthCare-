#!/usr/bin/env python
# coding: utf-8

# # Capstone Project- 
# ## Healthcare

# ### Problem Statement-
# 
# NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases) research creates knowledge about and treatments for the most chronic, costly, and consequential diseases.  
# 
# The dataset used in this project is originally from NIDDK. The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
# 
# 
# .Build a model to accurately predict whether the patients in the dataset have diabetes or not.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#import dataset
hc=pd.read_csv('health care diabetes.csv')


# In[3]:


hc.head()


# In[4]:


hc.shape


# In[5]:


hc.isnull().sum()


# In[6]:


hc.dtypes


# **1. Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:**
# 
# • **Glucose**
# 
# • **BloodPressure**
# 
# • **SkinThickness**
# 
# • **Insulin**
# 
# • **BMI**
# 
# **2. Visually explore these variables using histograms. Treat the missing values accordingly**.
# 
# **3. There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables.** 

# In[7]:


hc.describe()


# In[8]:


# finding zero values in all columns
for x in hc.columns[1:-1]:
    l=len(hc[hc[x]==0])
    if l>1:
        print(x,'  has {} zeros'.format(l))
    else:
        print(x,'  has no zero values')


# **A person cannot have zero values for glucose,bmi,blood pressure,insulin,skin thickness,diabetes pedigree function.all these zero values doesn't make sense so these are missing values so  we have to treat them accordingly.**

# In[9]:


hc_copy=hc.copy(deep=True)
hc_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=hc_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)


# In[10]:


hc_copy.isnull().sum()


# In[11]:


sns.pairplot(hc)


# In[12]:


diab=hc[hc['Outcome']==1]
diab.head()


# In[13]:


hc['Glucose'].value_counts().head(10)


# In[14]:


hc['Glucose'].plot(kind='hist',figsize=(5,6),title='Glucose')


# In[15]:


hc['BloodPressure'].value_counts().head()


# In[16]:


hc['BloodPressure'].plot(kind='hist',figsize=(5,6),title='BloodPressure')


# In[17]:


hc['SkinThickness'].value_counts().head(10)


# In[18]:


hc['SkinThickness'].plot(kind='hist',figsize=(5,6),title='SkinThickness')


# In[19]:


hc['Insulin'].value_counts().head()


# In[20]:


hc['Insulin'].plot(kind='hist',figsize=(5,6),title='Insulin')


# In[21]:


hc['BMI'].value_counts().head(10)


# In[22]:


hc['BMI'].plot(kind='hist',figsize=(5,6),title='BMI')


# ## Filling missing values-

# In[23]:


hc_copy.isnull().sum()


# In[24]:


#Replace missing values by Median
def median_val(var):
    med_hc=hc_copy[hc_copy[var].notnull()]
    med_hc=med_hc[[var,'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return med_hc


# In[25]:


median_val('Glucose')


# In[26]:


hc_copy.loc[(hc_copy['Outcome']==0)& hc_copy['Glucose'].isnull(),'Glucose']=107
hc_copy.loc[(hc_copy['Outcome']==1)& hc_copy['Glucose'].isnull(),'Glucose']=140


# In[27]:


median_val('BloodPressure')


# In[28]:


hc_copy.loc[(hc_copy['Outcome']==0)& hc_copy['BloodPressure'].isnull(),'BloodPressure']=70
hc_copy.loc[(hc_copy['Outcome']==1)& hc_copy['BloodPressure'].isnull(),'BloodPressure']=74.5


# In[29]:


median_val('SkinThickness')


# In[30]:


hc_copy.loc[(hc_copy['Outcome']==0)& hc_copy['SkinThickness'].isnull(),'SkinThickness']=27
hc_copy.loc[(hc_copy['Outcome']==1)& hc_copy['SkinThickness'].isnull(),'SkinThickness']=32


# In[31]:


median_val('Insulin')


# In[32]:


hc_copy.loc[(hc_copy['Outcome']==0)& hc_copy['Insulin'].isnull(),'Insulin']=102.5
hc_copy.loc[(hc_copy['Outcome']==1)& hc_copy['Insulin'].isnull(),'Insulin']=169.5


# In[33]:


median_val('BMI')


# In[34]:


hc_copy.loc[(hc_copy['Outcome']==0)& hc_copy['BMI'].isnull(),'BMI']=30.1
hc_copy.loc[(hc_copy['Outcome']==1)& hc_copy['BMI'].isnull(),'BMI']=34.3


# In[35]:


hc_copy.isnull().sum()


# In[36]:


#Pair plot after handling missing values
sns.pairplot(hc_copy)


# In[37]:


#Create a count (frequency) plot describing the data types and the count of variables.
sns.countplot(hc.dtypes.map(str),palette='Set2')


# ## Data Exploration
# 
# 1. Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.
# 
# 2. Create scatter charts between the pair of variables to understand the relationships. Describe your findings.
# 
# 3. Perform correlation analysis. Visually explore it using a heat map.

# In[38]:


hc.Outcome.value_counts()


# In[39]:


diab_count=hc.Outcome.astype('category').cat.rename_categories(['Healthy','diabetic'])


# In[40]:


sns.countplot(diab_count)


# In[41]:


sns.pairplot(hc_copy,hue='Outcome')


# ## Heatmap of original dataset

# In[42]:


hc.corr()


# In[43]:


plt.figure(figsize=(10,10))
sns.heatmap(hc.corr(),annot=True,cmap='viridis')


# ## Heatmap of Clean data

# In[44]:


plt.figure(figsize=(10,10))
sns.heatmap(hc_copy.corr(),annot=True,cmap='viridis')


# From the above heatmap we see a bit of correlation between some columns-.
# 
# 1.Age and Pregnancies = 0.54
# 
# 2.Glucose and insulin = 0.49
# 
# 3.SkinThickness and BMI = 0.57

# In[45]:


# let's visualise the relationship between above pairs
def pair(var1,var2):
    sns.scatterplot(x=var1,y=var2,data=hc_copy,hue='Outcome')


# In[46]:


pair('Age','Pregnancies')


# In[47]:


pair('Glucose','Insulin')


# In[48]:


pair('SkinThickness','BMI')


# ## Split the dataset

# In[90]:


x=hc_copy.iloc[:,:-1]
y=hc_copy.iloc[:,-1]


# In[91]:


x.shape,y.shape


# In[92]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)


# In[93]:


x_train.shape,y_train.shape,y_test.shape,x_test.shape


# In[94]:


#to bring whole data at a same scale then we will perform standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.transform(x_test)


# ## Perform different models to check accuracy

# # 1. LogisticRegression Model

# In[95]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train_scaled,y_train)


# In[96]:


log_pred=classifier.predict(x_test_scaled)


# In[97]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[98]:


print(classification_report(y_test,log_pred))


# In[99]:


print(confusion_matrix(y_test,log_pred))


# In[100]:


print('\n','Accuracy-',accuracy_score(y_test,log_pred))


# ## Random forest classifier

# In[101]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc=RandomForestClassifier()


# In[102]:


n_estimators=[75,100,125,150,200]
max_features=[4,5,6,7,8]
bootstrap=[True,False]
oob_score=[True,False]


# In[103]:


param_grid={'n_estimators':n_estimators,
            'max_features':max_features,
             'bootstrap':bootstrap,
              'oob_score':oob_score}


# In[104]:


grid=GridSearchCV(rfc,param_grid)


# In[108]:


import warnings
warnings.filterwarnings('ignore')


# In[109]:


grid.fit(x_train_scaled,y_train)


# In[110]:


rfc_pred=grid.predict(x_test_scaled)


# In[111]:


print(classification_report(y_test,rfc_pred))


# In[112]:


print(confusion_matrix(y_test,log_pred))
print('\n','Accuracy',accuracy_score(y_test,rfc_pred))


# In[114]:


grid.best_params_


# ## Support Vector Machine

# In[115]:


from sklearn.svm import SVC


# In[116]:


svc=SVC()


# In[117]:


param_grid = {'C':[0.01,0.1,1,10],'kernel':['linear', 'poly', 'rbf', 'sigmoid']}


# In[118]:


grid=GridSearchCV(svc,param_grid)


# In[119]:


grid.fit(x_train_scaled,y_train)


# In[121]:


svc_predict=grid.predict(x_test_scaled)


# In[122]:


print(classification_report(y_test,svc_predict))


# In[123]:


print(confusion_matrix(y_test,log_pred))
print('\n','Accuracy',accuracy_score(y_test,svc_predict))


# In[124]:


grid.best_params_


# ## 4. K Nearest Neighbour

# In[130]:


from sklearn.neighbors import KNeighborsClassifier
train_score=[]
test_score=[]
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_scaled,y_train)
    train_score.append(knn.score(x_train_scaled,y_train))
    test_score.append(knn.score(x_test_scaled,y_test))


# In[131]:


sns.lineplot(x=range(1,20),y=train_score,marker='o')
sns.lineplot(x=range(1,20),y=test_score,marker='o')


# In[137]:


acc_score=[]
for i in range(1,10):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_scaled,y_train)
    knn_pred=knn.predict(x_test_scaled)
    acc_score.append(accuracy_score(y_test,knn_pred))
print(max(acc_score))


# In[138]:


sns.lineplot(x=range(1,10),y=acc_score,marker='o')


# **From the above results,n=7 gives the best results so we will take n_neighbors=7 for final  model**

# In[141]:


knn_model=KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train_scaled,y_train)


# In[142]:


knn_pred=knn_model.predict(x_test_scaled)


# In[145]:


print(classification_report(y_test,knn_pred))


# In[147]:


print(confusion_matrix(y_test,knn_pred))
print('\n','Accuracy',accuracy_score(y_test,knn_pred))


# ## 5. Decision Tree

# In[148]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=50)
dtc.fit(x_train_scaled,y_train)


# In[149]:


dtc_pred=dtc.predict(x_test_scaled)


# In[151]:


print(classification_report(y_test,dtc_pred))


# In[152]:


print(confusion_matrix(y_test,dtc_pred))


# In[153]:


print(accuracy_score(y_test,dtc_pred))


# ## Accuracy,Sensitivity,Specificity

# In[156]:


models=[
    {'label':'LogisticRegression',
      'model':LogisticRegression(),
    },
    {'label':'RandomForestClassifier',
      'model':RandomForestClassifier(),
    },
    {'label': 'KNeighbors Classifier',
    'model': KNeighborsClassifier(n_neighbors=7),
},
{
    'label' : 'Support Vector Classifier',
    'model' : SVC(C= 1, kernel='rbf',probability=True),
},
{
    'label' : 'Decision Tress',
    'model' : DecisionTreeClassifier(random_state=42,criterion= 'entropy', max_features= 6, min_samples_split= 3),
},
    
]


# In[160]:


accu=[]
model_name=[]
sensitivity=[]
specificity=[]
for m in models:
    mod_1=m['model']
    mod_1.fit(x_train_scaled,y_train)
    mod_pred=mod_1.predict(x_test_scaled)
    cm=confusion_matrix(y_test,mod_pred)
    
    accu.append(accuracy_score(y_test,mod_pred))
    model_name.append(m['label'])
        
    sensitivity.append(cm[0,0]/(cm[0,0]+cm[0,1]))
    specificity.append(cm[1,1]/(cm[1,0]+cm[1,1]))
    
    models_accu_sen_sp= pd.DataFrame(data=(accu,sensitivity,specificity),index = ['Accuracy','Sensitivity','Specificity'],
                                    columns=[model_name]).T
models_accu_sen_sp    


# ## ROC curve

# In[164]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# Below for loop iterates through your models list
for m in models:
    model = m['model'] # select the model
    model.fit(x_train_scaled, y_train) # train the model
    y_pred=model.predict(x_test_scaled) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test_scaled)[:,1])
# Calculate Area under the curve to display on the plot
    auc = roc_auc_score(y_test,model.predict(x_test_scaled))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc =(1.01,0))
plt.show()   # Display


# Random forest classifier has the best curve area so this is the best model for this problem.

# In[ ]:




