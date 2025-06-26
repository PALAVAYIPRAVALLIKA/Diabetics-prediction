#!/usr/bin/env python
# coding: utf-8

# In[248]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action='ignore')
sns.set()
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[249]:


# Load the dataset
df = pd.read_csv("C:/Users/abign/Downloads/diabetics data set.csv")


# In[250]:


df.head()


# In[251]:


df.info()


# In[252]:


df.columns


# In[253]:


df.describe()


# In[254]:


# (row, columns)
df.shape


# In[255]:


# distribution of outcome variable
df.Outcome.value_counts()*100/len(df)


# In[256]:


df['Outcome'].value_counts()*100/len(df)


# In[257]:


# plot the hist of the age variable
plt.figure(figsize=(8,7))
plt.xlabel('Age', fontsize=10)
plt.ylabel('Count', fontsize=10)
df['Age'].hist(edgecolor="black")


# In[258]:


df['Age'].max()


# In[259]:


df['Age'].min()


# In[260]:


print("MAX AGE: "+str(df['Age'].max()))
print("MIN AGE: "+str(df['Age'].min()))


# In[261]:


df.columns


# In[262]:


# density graph
fig,ax = plt.subplots(4,2, figsize=(20,20))
sns.distplot(df.Pregnancies, bins=20, ax=ax[0,0], color="red")
sns.distplot(df.Glucose, bins=20, ax=ax[0,1], color="red")
sns.distplot(df.BloodPressure, bins=20, ax=ax[1,0], color="red")
sns.distplot(df.SkinThickness, bins=20, ax=ax[1,1], color="red")
sns.distplot(df.Insulin, bins=20, ax=ax[2,0], color="red")
sns.distplot(df.BMI, bins=20, ax=ax[2,1], color="red")
sns.distplot(df.DiabetesPedigreeFunction, bins=20, ax=ax[3,0], color="red")
sns.distplot(df.Age, bins=20, ax=ax[3,1], color="red")


# In[263]:


df.columns


# In[264]:


df.groupby("Outcome").agg({'Pregnancies':'mean'})


# In[265]:


df.groupby("Outcome").agg({'Pregnancies':'max'})


# In[266]:


df.groupby("Outcome").agg({'Glucose':'mean'})


# In[267]:


df.groupby("Outcome").agg({'Glucose':'max'})


# In[268]:


# 0>healthy
# 1>diabetes

#f,ax = plt.subplots(1,2, figsize=(18,8))
#df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct = "%1.1f%%", ax=ax[0], shadow=True)
#ax[0].set_title('target')
#ax[0].set_ylabel('')
#sns.countplot('Outcome', data=df, ax=ax[1])
#ax[1].set_title('Outcome')
#plt.show()


# In[269]:


f,ax = plt.subplots(figsize=[20,15])
sns.heatmap(df.corr(), annot=True, fmt = '.2f', ax=ax, cmap='magma')
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# In[270]:


df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)


# In[271]:


# Data preprocessing Part
df.isnull().sum()


# In[272]:


#import missingno as msno
#msno.bar(df, color="orange")


# In[273]:


#median
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# In[274]:


columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]


# In[275]:


df.isnull().sum()


# In[276]:


# pair plot
p = sns.pairplot(df, hue="Outcome")


# In[277]:


# Outlier Detection
# IQR+Q1
# 50%
# 24.65->25%+50%
# 24.65->25%
for feature in df:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    if df[(df[feature]>upper)].any(axis=None):
        print(feature, "yes")
    else:
        print(feature, "no")


# In[278]:


plt.figure(figsize=(8,7))
sns.boxplot(x= df["Insulin"], color="red")


# In[279]:


Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR
df.loc[df['Insulin']>upper, "Insulin"] = upper
plt.figure(figsize=(8,7))
sns.boxplot(x= df["Insulin"], color="red")


# In[280]:


# LOF
# local outlier factor
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=10)
lof.fit_predict(df)


# In[281]:


plt.figure(figsize=(8,7))
sns.boxplot(x= df["Pregnancies"], color="red")


# In[282]:


df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:20]


# In[283]:


thresold = np.sort(df_scores)[7]
thresold


# In[284]:


outlier = df_scores>thresold
df = df[outlier]


# In[285]:


plt.figure(figsize=(8,7))
sns.boxplot(x= df["Pregnancies"], color="red")


# In[286]:


# Feature Enginnering
NewBMI = pd.Series(["Underweight","Normal", "Overweight","Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
NewBMI


# In[287]:


df['NewBMI'] = NewBMI
df.loc[df["BMI"]<18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"]>18.5) & df["BMI"]<=24.9, "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"]>24.9) & df["BMI"]<=29.9, "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"]>29.9) & df["BMI"]<=34.9, "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"]>34.9) & df["BMI"]<=39.9, "NewBMI"] = NewBMI[4]
df.loc[df["BMI"]>39.9, "NewBMI"] = NewBMI[5]


# In[288]:


# if insulin>=16 & insuline<=166->normal
def set_insuline(row):
    if row["Insulin"]>=16 and row["Insulin"]<=166:
        return "Normal"
    else:
        return "Abnormal"
    
df = df.assign(NewInsulinScore=df.apply(set_insuline, axis=1))


# In[289]:


# Some intervals were determined according to the glucose variable and these were assigned categorical variables.
NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")
df["NewGlucose"] = NewGlucose
df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]


# In[290]:


# One hot encoding
df = pd.get_dummies(df, columns = ["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)


# In[291]:


categorical_df = df[['NewBMI_Obesity 1',
       'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
       'NewBMI_Underweight', 'NewInsulinScore_Normal', 'NewGlucose_Low',
       'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]
categorical_df.head()


# In[292]:


y=df['Outcome']
X=df.drop(['Outcome','NewBMI_Obesity 1',
       'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
       'NewBMI_Underweight', 'NewInsulinScore_Normal', 'NewGlucose_Low',
       'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis=1)
cols = X.columns
index = X.index


# In[293]:


from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X=transformer.transform(X)
X=pd.DataFrame(X, columns = cols, index = index)
X = pd.concat([X, categorical_df], axis=1)


# In[294]:


X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[295]:


scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[296]:


# Machine Learning Algorithms
# Logistic Regreesion

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[297]:


# Machine Learning Algorithms
# Logistic Regression
# Machine Learning Algorithms
# Logistic Regression



y_pred = log_reg.predict(X_test)
accuracy_score(y_train, log_reg.predict(X_train))



log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

#  ROC-AUC score calculation
print("ROC-AUC Score:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]))


# In[298]:


# SVM

svc = SVC(probability=True)
parameter = {
    "gamma": [0.0001, 0.001, 0.01, 0.1],
    'C': [0.01, 0.05, 0.5, 1, 10, 15, 20]  
}
grid_search = GridSearchCV(svc, parameter)
grid_search.fit(X_train, y_train)
grid_search.best_params_


{'C': 10, 'gamma': 0.01}


grid_search.best_score_




svc = SVC(C=10, gamma=0.01, probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, svc.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#  ROC-AUC score calculation
print("\nROC-AUC Score:", roc_auc_score(y_test, svc.predict_proba(X_test)[:, 1]))
   


# In[299]:


# random forest
rand_clf = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75,
                                min_samples_leaf=2, min_samples_split=3,
                                n_estimators=130)
rand_clf.fit(X_train, y_train)

# Output from model initialization
RandomForestClassifier
RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75,
min_samples_leaf=2, min_samples_split=3,
n_estimators=130)

# Model evaluation
y_pred = rand_clf.predict(X_test)
print(accuracy_score(y_train, rand_clf.predict(X_train)))
rand_acc = accuracy_score(y_test, rand_clf.predict(X_test))
print(accuracy_score(y_test, rand_clf.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#  ROC-AUC score calculation
print("ROC-AUC Score:", roc_auc_score(y_test, rand_clf.predict_proba(X_test)[:, 1]))

  


# In[301]:


# Model Comparison
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest Classifier'], 
    'Score': [100*round(log_reg_acc,4),100*round(svc_acc,4), 100*round(rand_acc,4)]
})
models.sort_values(by = 'Score', ascending = False)


# In[ ]:




