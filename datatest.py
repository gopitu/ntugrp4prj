#!/usr/bin/env python
# coding: utf-8

# In[143]:


import pandas as pd


# In[144]:


df_test = pd.read_csv('fraudTest.csv')


# In[145]:


len(df_test)


# In[146]:


df_test = df_test.drop(df_test.columns[0], axis='columns')


# In[147]:


df_test = df_test.drop(['trans_date_trans_time', 'merchant', 'first', 'last', 'street','city','state','job','dob','trans_num'], axis='columns')


# In[148]:


df_test.info()


# In[149]:


from sklearn.model_selection import train_test_split


# In[150]:


X = df_test.drop('is_fraud', axis=1)


# In[151]:


cols = ['cc_num','category','amt','gender','zip','lat','long','city_pop','unix_time','merch_lat','merch_long']
X = pd.get_dummies(X[cols])


# In[152]:


X.info()


# In[175]:


X.to_csv('normalized_x.csv', index=False)


# In[153]:


Y = df_test[['is_fraud']]


# In[154]:


Y.info()


# In[155]:


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[156]:


from sklearn.tree import DecisionTreeClassifier


# In[157]:


model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)


# In[158]:


model.fit(X_Train, Y_Train)


# In[160]:


y_predict = model.predict(X_Test)


# In[161]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[162]:


accuracy_score(Y_Test, y_predict)


# In[163]:


cm = confusion_matrix(Y_Test, y_predict)


# In[164]:


cm


# In[165]:


import joblib


# In[166]:


joblib.dump(model, 'group_prj_cc_fraud')


# In[168]:


from sklearn.tree import export_text
feature_names = list(X.columns)
r = export_text(model, feature_names=feature_names)
print(r)


# In[171]:


from sklearn import tree


# In[173]:


tree.plot_tree(model)


# In[ ]:





# In[ ]:




