#!/usr/bin/env python
# coding: utf-8

# In[152]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn


# In[153]:


df=pd.read_csv(r'C:\Users\Ved prakash\Desktop\FilemNameranges1log.csv')


# In[154]:


df


# In[155]:


df2=df.drop('Unnamed: 0',axis=1)


# In[156]:


df2


# In[157]:


df3=df2.drop('logtargect',axis=1)


# In[158]:


df3


# In[159]:


Y=df3["Standard Value"]


# In[160]:


Y


# In[161]:


X= df3.drop("Standard Value",axis=1)


# In[162]:


from sklearn.feature_selection import VarianceThreshold
 
def remove_low_variance(df3, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(df3)
    return df3[df3.columns[selection.get_support(indices=True)]]
 
X = remove_low_variance(df3, threshold=0.1)
X


# In[111]:


from sklearn.preprocessing import StandardScaler


# In[112]:


Scaler =StandardScaler()
Scaler.fit(X)


# In[113]:


Scaled_data = Scaler.transform(X)


# In[114]:


from sklearn.decomposition import PCA


# In[115]:


pca=PCA(n_components=0.99)


# In[116]:


pca.fit(Scaled_data)


# In[117]:


X_pca = pca.transform(Scaled_data)


# In[118]:


Scaled_data.shape


# In[119]:


X_pca.shape


# In[163]:


plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=Y)


# In[ ]:





# In[193]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=42)


# In[194]:


from sklearn.preprocessing import MinMaxScaler


# In[195]:


scaler = MinMaxScaler()


# In[196]:


scaler.fit(X_train)


# In[197]:


X_train= scaler.transform(X_train)


# In[198]:


X_test= scaler.transform(X_test)


# In[199]:


X_train.max()


# In[200]:


Y_train.max()


# In[ ]:





# In[201]:


from sklearn.linear_model import LinearRegression


# In[202]:


model5 = LinearRegression()


# In[203]:


model5.fit(X_train,Y_train)


# In[204]:


model5.score(X_test,Y_test)


# In[205]:


from sklearn.ensemble import RandomForestRegressor


# In[177]:


model3 = RandomForestRegressor()


# In[178]:


model3.fit(X_train,Y_train)


# In[179]:


model3.score(X_test,Y_test)


# In[180]:


from sklearn import preprocessing
from sklearn import utils


# In[39]:


lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y)
print (training_scores_encoded)
print (utils.multiclass.type_of_target(Y))
print (utils.multiclass.type_of_target(Y.astype('int')))
print (utils.multiclass.type_of_target(training_scores_encoded))


# In[ ]:





# In[40]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[41]:



from sklearn.svm import SVC


# In[42]:


from sklearn import svm
clf = SVC()


# In[43]:


clf.fit(X_train,training_scores_encoded)


# In[ ]:


clf

