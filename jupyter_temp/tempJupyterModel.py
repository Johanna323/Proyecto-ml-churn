#!/usr/bin/env python
# coding: utf-8

# # Regresión lógística

# ## Carga de datos

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df_data = pd.read_csv('./Telco-Customer-Churn.csv')


# In[3]:


df_data.head(5)


# In[4]:


df_data.info()


# ## Cambiar total_charges a numeric

# In[5]:


df_data.TotalCharges = pd.to_numeric(df_data.TotalCharges, errors='coerce')


# ## Manejo de datos nulos

# In[6]:


df_data.isnull().sum()


# In[8]:


df_data.dropna(inplace=True)


# ## Eliminar id

# In[10]:


df_data.head(5)


# In[11]:


df_data.drop('customerID',axis=1,inplace=True)


# ## Convertir a numérico variable objetivo

# In[12]:


df_data['Churn'].replace(to_replace='Yes', value = 1, inplace=True)
df_data['Churn'].replace(to_replace='No', value = 0, inplace=True)


# In[13]:


df_data_processing = df_data.copy()


# ## Manejo de variables categóricas

# In[14]:


df_data_processing = pd.get_dummies(df_data_processing)
df_data_processing.head(5)


# ## Analisis de correlación

# In[15]:


import matplotlib.pyplot as plt


# In[16]:


fig = plt.figure(figsize=(15,9))
df_data_processing.corr()['Churn'].sort_values(ascending=True).plot(kind='bar')
plt.show()


# ## Escalabilidad de los datos

# In[17]:


from sklearn.preprocessing import MinMaxScaler


# In[18]:


scaler = MinMaxScaler()
df_data_processing_scaled =  scaler.fit_transform(df_data_processing)


# In[19]:


df_data_processing_scaled = pd.DataFrame(df_data_processing_scaled)


# In[20]:


df_data_processing_scaled.columns = df_data_processing.columns


# In[21]:


df_data_processing_scaled.head(5)


# ## Análisis exploratorio de datos

# In[22]:


import seaborn as sns


# In[23]:


sns.countplot(data=df_data, x='gender',hue='Churn')
plt.show()


# In[24]:


def plot_categorial(column):
    fig = plt.figure(figsize=(10,10))
    sns.countplot(data=df_data, x=column,hue='Churn')
    plt.show()


# In[25]:


column_cat = df_data.select_dtypes(include='object').columns


# In[26]:


for _ in column_cat:
    plot_categorial(_)


# In[27]:


fig = plt.figure(figsize=(10,10))
sns.pairplot(data= df_data, hue='Churn')
plt.show()


# ## Entrenamiento del modelo de regresión logística binomial

# #TIPOS DE MODELOS DE CLASIFICACION
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# 
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "SVM": SVC()
# }

# In[28]:


X = df_data_processing_scaled.drop('Churn',axis=1)
y = df_data_processing_scaled['Churn'].values


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[30]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[31]:


from sklearn import metrics
prediction_test = model.predict(X_test)
print(metrics.accuracy_score(y_test,prediction_test ))


# ## Evaluación del modelo

# In[32]:


model.predict_proba(X_test)


# In[33]:


model.coef_


# In[34]:


model.feature_names_in_


# In[35]:


weights = pd.Series(model.coef_[0],
                    index=X.columns.values) 
print(weights.sort_values(ascending=False)[:10].plot(kind='bar'))


# In[36]:


print(weights.sort_values(ascending=False)[-10:].plot(kind='bar'))


# In[37]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[38]:


fig = plt.figure(figsize=(11,11))
cm = confusion_matrix(y_test, prediction_test, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=model.classes_)
disp.plot(cmap='gray')
plt.show()


# In[40]:


print("Accuracy model: ", metrics.accuracy_score(y_test, prediction_test))
print("Recall Churn='NO': ", metrics.recall_score(y_test, prediction_test, pos_label=0))
print("Recall Churn='YES': ", metrics.recall_score(y_test, prediction_test, pos_label=1))


# In[42]:


import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train_sm)
results = model.fit()

print(results.summary())


# In[43]:


df_data2 = df_data.copy()
df_data2 = df_data2.drop(['gender', 'MultipleLines', 'PhoneService', 'StreamingMovies', 'StreamingTV'], axis=1)


# In[44]:


df_data2 = pd.get_dummies(df_data2)
df_scaled_r = scaler.fit_transform(df_data2)
df_scaled_r = pd.DataFrame(df_scaled_r)
df_scaled_r.columns = df_data2.columns


# In[45]:


x_r = df_scaled_r.drop(['Churn'], axis=1)
y_r = df_scaled_r['Churn']


# In[48]:


#Opcional
from imblearn.over_sampling import SMOTE
oversample_r = SMOTE()
x_rsmote, y_rsmote = oversample_r.fit_resample(x_r, y_r)


# In[49]:


x_train_rsmote, x_test_rsmote, y_train_rsmote, y_test_rsmote = train_test_split(x_rsmote,y_rsmote,train_size=0.7, random_state=42)


# In[50]:


model_rsmote = LogisticRegression()
model_rsmote.fit(x_train_rsmote, y_train_rsmote)
pred_rsmote = model_rsmote.predict(x_test_rsmote)


# In[51]:


cm_rsmote = confusion_matrix(y_test_rsmote, pred_rsmote, labels=model_rsmote.classes_)
disp_rsmote = ConfusionMatrixDisplay(confusion_matrix=cm_rsmote, display_labels=model_rsmote.classes_)
disp_rsmote.plot(cmap='gray')
plt.show()
# In[52]:
print("Accuracy model: ", metrics.accuracy_score(y_test_rsmote, pred_rsmote))
print("Recall Churn='NO': ", metrics.recall_score(y_test_rsmote, pred_rsmote, pos_label=0))
print("Recall Churn='YES': ", metrics.recall_score(y_test_rsmote, pred_rsmote, pos_label=1))