#!/usr/bin/env python
# coding: utf-8

# In[302]:


import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors (2).csv')


# In[3]:


df


# In[246]:


df2=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors2.csv')


# In[248]:


df2


# In[ ]:





# In[ ]:





# In[249]:


df3=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors3.csv')


# In[250]:


df3


# In[ ]:





# In[251]:


df4=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors4.csv')


# In[252]:


df4


# In[ ]:





# In[253]:


df5=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors5.csv')


# In[254]:


df5


# In[ ]:





# In[255]:


df6=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors6.csv')


# In[256]:


df6


# In[ ]:





# In[257]:


df7=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors7.csv')


# In[258]:


df7


# In[ ]:





# In[259]:


df8=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors8.csv')


# In[260]:


df8


# In[ ]:





# In[261]:


df9=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors9.csv')


# In[262]:


df9


# In[ ]:





# In[263]:


df10=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors10.csv')


# In[264]:


df10


# In[ ]:





# In[ ]:





# In[265]:


df11=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors11.csv')


# In[266]:


df11


# In[ ]:





# In[267]:


df12=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors12.csv')


# In[268]:


df12


# In[ ]:





# In[269]:


df13=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors13.csv')


# In[270]:


df13


# In[ ]:





# In[271]:


df14=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors14.csv')


# In[272]:


df14


# In[ ]:





# In[273]:


df15=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors15.csv')


# In[274]:


df15


# In[ ]:





# In[275]:


df16=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors16.csv')


# In[276]:


df16


# In[ ]:





# In[277]:


df17=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors17.csv')


# In[278]:


df17


# In[ ]:





# In[279]:


df18=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors18.csv')


# In[280]:


df18


# In[ ]:





# In[281]:


df19=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors19.csv')


# In[282]:


df19


# In[ ]:





# In[283]:


df20=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors20.csv')


# In[284]:


df20


# In[ ]:





# In[285]:


df21=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors21.csv')


# In[286]:


df21


# In[ ]:





# In[287]:


df22=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors22.csv')


# In[288]:


df22


# In[ ]:





# In[289]:


df23=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors23.csv')


# In[290]:


df23


# In[ ]:





# In[291]:


df24=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors24.csv')


# In[292]:


df24


# In[ ]:





# In[293]:


df25=pd.read_csv(r'C:\Users\Ved prakash\Downloads\descriptors25.csv')


# In[294]:


df25


# In[ ]:





# In[28]:


df2


# In[ ]:





# In[295]:


rames=[df,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25]


# In[ ]:





# In[296]:


result2=pd.concat(rames)


# In[34]:


result2


# In[303]:


dff=pd.read_excel(r'C:\Users\Ved prakash\Desktop\afterdropeanaip.xlsx')


# In[327]:


dff


# In[334]:


dff.drop('Name', axis=1,inplace=True)


# In[335]:


dff.isnull().sum()


# In[336]:


from sklearn.feature_selection import VarianceThreshold


# In[337]:


var_thres=VarianceThreshold(threshold=0)


# In[338]:


var_thres.fit(dff)


# In[326]:


dff


# In[347]:


dfnt=pd.read_excel('C:\\Users\\Ved prakash\\desktop\\a2kkkk.xlsx')


# In[348]:


dfnt.dtypes


# In[353]:


dfnt["Standard Value"]


# In[354]:


Y


# In[356]:


X=dfnt


# In[357]:


X


# In[342]:


from sklearn.model_selection import train_test_split
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


# In[100]:


input1=dff1['Smiles']


# In[149]:


list1 = []
for i in range(225):
  list1.append(input1[i])


# In[150]:


list1


# In[104]:


pip install padelpy


# In[ ]:


from padelpy import from_smiles


# In[123]:


descriptors1 = from_smiles(['Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3cccc(N)c3)c(=O)c21',
 'COc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)cc1',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(C(F)(F)F)cc3)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(F)c3F)c(=O)c1n2C',
 'Cn1cc(C(=O)c2cccc(Cl)c2Cl)cc1C(=O)NCNC(=O)c1cc(C(=O)c2cccc(Cl)c2Cl)cn1C',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(F)cc3)c(=O)c1n2C',
 'CCCCCn1ncc2c3sc(C)cc3n(C)c2c1=O',
 'Cn1c2cc(C#N)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccc2ccccc2c1'
 ])
_= from_smiles([
 'Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3cccc(N)c3)c(=O)c21',
 'COc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)cc1',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(C(F)(F)F)cc3)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(F)c3F)c(=O)c1n2C',
 'Cn1cc(C(=O)c2cccc(Cl)c2Cl)cc1C(=O)NCNC(=O)c1cc(C(=O)c2cccc(Cl)c2Cl)cn1C',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(F)cc3)c(=O)c1n2C',
 'CCCCCn1ncc2c3sc(C)cc3n(C)c2c1=O',
 'Cn1c2cc(C#N)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccc2ccccc2c1'], output_csv='descriptors1.csv')


# In[124]:


import os.path,time
print("last modified:%s"%time.ctime(os.path.getmtime("descriptors1.csv")))
print("created:%s"%time.ctime(os.path.getctime("descriptors1.csv")))


# In[125]:


descriptors2 = from_smiles(['O=S(=O)(Cc1ccccc1)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'O=S(=O)(Nc1ccc(-c2csc(=S)n2Cc2cccnc2)cc1)c1cccc2cccnc12',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2ccncc2)cc1)c1ccc2c(c1)OCCO2',
 'Cc1ccncc1CN1C(=S)SCC1(O)c1ccc(NS(=O)(=O)c2ccc3c(c2)OCCO3)cc1',
 'Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3cccc(O)c3)c(=O)c21',
 'COc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)cc1F',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(F)c(F)c3F)c(=O)c1n2C',
 'Cc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)cc1',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(Cl)c3)c(=O)c1n2C',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1',
 'O=S(=O)(NCC1CN(S(=O)(=O)c2c(F)cccc2F)C1)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1ccc2c(c1)OCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'O=S(=O)(c1c(F)cccc1F)C1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2ccccn2)CC1',
 'O=S(=O)(c1ccccc1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'COc1cccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c1'])
_= from_smiles(['O=S(=O)(Cc1ccccc1)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'O=S(=O)(Nc1ccc(-c2csc(=S)n2Cc2cccnc2)cc1)c1cccc2cccnc12',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2ccncc2)cc1)c1ccc2c(c1)OCCO2',
 'Cc1ccncc1CN1C(=S)SCC1(O)c1ccc(NS(=O)(=O)c2ccc3c(c2)OCCO3)cc1',
 'Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3cccc(O)c3)c(=O)c21',
 'COc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)cc1F',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(F)c(F)c3F)c(=O)c1n2C',
 'Cc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)cc1',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(Cl)c3)c(=O)c1n2C',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1',
 'O=S(=O)(NCC1CN(S(=O)(=O)c2c(F)cccc2F)C1)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1ccc2c(c1)OCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'O=S(=O)(c1c(F)cccc1F)C1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2ccccn2)CC1',
 'O=S(=O)(c1ccccc1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'COc1cccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c1'],output_csv='descriptors2.csv')


# In[126]:


import os.path,time
print("last modified:%s"%time.ctime(os.path.getmtime("descriptors2.csv")))
print("created:%s"%time.ctime(os.path.getctime("descriptors2.csv")))


# In[127]:


descriptors3 = from_smiles(['O=C(CCSC(=S)NCc1cccnc1)c1cnn2ncccc12',
 'CCCCN(Cc1ccccc1)C(=O)c1cc(Br)c[nH]1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1cccc2ccccc12',
 'O=C(CCSC(=S)NCc1cccnc1)c1cn(CC(F)F)c2ncccc12',
 'C#Cc1ccnc2[nH]cc(C(=O)CCSC(=S)NCc3cccnc3)c12',
 'O=C(CCSC(=S)NCc1cccnc1)c1cn(-c2cccs2)c2ncccc12',
 'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2ncccc12',
 'Cc1cccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)c1',
 'N#Cc1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'Cc1ccccc1S(=O)(=O)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'Cc1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'C=C1C(=O)O[C@H]2[C@H]1CC/C(C(=O)OCCOCCOC(=O)/C1=C/CC[C@@]3(C)O[C@H]3[C@H]3OC(=O)C(=C)[C@@H]3CC1)=C\\CC[C@@]1(C)O[C@@H]21'])
_= from_smiles(['O=C(CCSC(=S)NCc1cccnc1)c1cnn2ncccc12',
 'CCCCN(Cc1ccccc1)C(=O)c1cc(Br)c[nH]1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1cccc2ccccc12',
 'O=C(CCSC(=S)NCc1cccnc1)c1cn(CC(F)F)c2ncccc12',
 'C#Cc1ccnc2[nH]cc(C(=O)CCSC(=S)NCc3cccnc3)c12',
 'O=C(CCSC(=S)NCc1cccnc1)c1cn(-c2cccs2)c2ncccc12',
 'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2ncccc12',
 'Cc1cccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)c1',
 'N#Cc1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'Cc1ccccc1S(=O)(=O)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'Cc1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'C=C1C(=O)O[C@H]2[C@H]1CC/C(C(=O)OCCOCCOC(=O)/C1=C/CC[C@@]3(C)O[C@H]3[C@H]3OC(=O)C(=C)[C@@H]3CC1)=C\\CC[C@@]1(C)O[C@@H]21'],output_csv='descriptors3.csv')


# In[128]:


import os.path,time
print("last modified:%s"%time.ctime(os.path.getmtime("descriptors3.csv")))
print("created:%s"%time.ctime(os.path.getctime("descriptors3.csv")))


# In[130]:


descriptors4 = from_smiles(['Cn1c2cc(CO)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'CCc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'O=S(=O)(c1ccc2ccccc2c1)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'N#Cc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1',
 'Cc1cc2c(s1)c1cnn(-c3ccccc3)c(=O)c1n2C',
 'CSc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'C[C@H]1CN(S(=O)(=O)c2c(F)cccc2F)CCN1S(=O)(=O)c1ccc2c(c1)OCCO2',
 'Nc1cccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c1',
 'O=C1CN(S(=O)(=O)c2c(F)cccc2F)CCN1S(=O)(=O)c1ccc2c(c1)OCCO2',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(F)cc3F)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(Cl)cc3)c(=O)c1n2C',
 'CCn1c2cc(C)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'CC(O)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'Cn1c2cc(B(O)O)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(F)c3)c(=O)c1n2C',
 'O=S(=O)(NC1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1)c1c(F)cccc1F',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'O=S(=O)(c1cccc(C(F)(F)F)c1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2cccc[n+]2[O-])CC1',
 'CCCc1cc(F)c(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c(F)c1'])
_= from_smiles(['Cn1c2cc(CO)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'CCc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'O=S(=O)(c1ccc2ccccc2c1)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'N#Cc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1',
 'Cc1cc2c(s1)c1cnn(-c3ccccc3)c(=O)c1n2C',
 'CSc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'C[C@H]1CN(S(=O)(=O)c2c(F)cccc2F)CCN1S(=O)(=O)c1ccc2c(c1)OCCO2',
 'Nc1cccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c1',
 'O=C1CN(S(=O)(=O)c2c(F)cccc2F)CCN1S(=O)(=O)c1ccc2c(c1)OCCO2',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(F)cc3F)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(Cl)cc3)c(=O)c1n2C',
 'CCn1c2cc(C)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'CC(O)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'Cn1c2cc(B(O)O)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(F)c3)c(=O)c1n2C',
 'O=S(=O)(NC1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1)c1c(F)cccc1F',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'O=S(=O)(c1cccc(C(F)(F)F)c1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2cccc[n+]2[O-])CC1',
 'CCCc1cc(F)c(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c(F)c1'],output_csv='descriptors4.csv')


# In[131]:


import os.path,time
print("last modified:%s"%time.ctime(os.path.getmtime("descriptors4.csv")))
print("created:%s"%time.ctime(os.path.getctime("descriptors4.csv")))


# In[133]:


descriptors5 = from_smiles(['O=S(=O)(NCCCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
'C[C@H]1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)CCN1S(=O)(=O)c1c(F)cccc1F',
'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2c(F)ccc(O)c2F)CC1',
'O=S(=O)(NC1CCN(S(=O)(=O)c2c(F)cccc2F)C1)c1ccc2c(c1)OCCO2',
'COc1cc(F)c(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c(F)c1',
'COc1cc(F)c(S(=O)(=O)N2CCCN(S(=O)(=O)c3cccc(N)c3)CC2)c(F)c1',
'Nc1cc(C(=O)N(Cc2ccc(O)cc2F)CC2CCC2)[nH]n1',
'O=C(c1cc(Cl)n[nH]1)N(Cc1cccc2[nH]ccc12)CC1CCC1',
'O=C(NCc1ccccc1)c1cc(Br)c[nH]1',
'COC(=O)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
'CC(=O)Nc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
'Cc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
'Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3cccc(N)c3)c(=O)c21',
'O=S(=O)(c1ccc2c(c1)OCCCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
'O=C(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
'Nc1cccc(S(=O)(=O)N2CCCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c1'])
_= from_smiles(['O=S(=O)(NCCCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
'C[C@H]1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)CCN1S(=O)(=O)c1c(F)cccc1F',
'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2c(F)ccc(O)c2F)CC1',
'O=S(=O)(NC1CCN(S(=O)(=O)c2c(F)cccc2F)C1)c1ccc2c(c1)OCCO2',
'COc1cc(F)c(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c(F)c1',
'COc1cc(F)c(S(=O)(=O)N2CCCN(S(=O)(=O)c3cccc(N)c3)CC2)c(F)c1',
'Nc1cc(C(=O)N(Cc2ccc(O)cc2F)CC2CCC2)[nH]n1',
'O=C(c1cc(Cl)n[nH]1)N(Cc1cccc2[nH]ccc12)CC1CCC1',
'O=C(NCc1ccccc1)c1cc(Br)c[nH]1',
'COC(=O)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
'CC(=O)Nc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
'Cc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
'Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3cccc(N)c3)c(=O)c21',
'O=S(=O)(c1ccc2c(c1)OCCCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
'O=C(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
'Nc1cccc(S(=O)(=O)N2CCCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)c1'],output_csv='descriptors5.csv')


# In[135]:


import os.path,time
print("last modified:%s"%time.ctime(os.path.getmtime("descriptors5.csv")))
print("created:%s"%time.ctime(os.path.getctime("descriptors5.csv")))


# In[223]:


dfm1=pd.read_csv(r'C:\Users\Ved prakash\descriptors.csv')


# In[224]:


dfm2=pd.read_csv(r'C:\Users\Ved prakash\descriptors1.csv')


# In[225]:


dfm3=pd.read_csv(r'C:\Users\Ved prakash\descriptors2.csv')


# In[226]:


dfm4=pd.read_csv(r'C:\Users\Ved prakash\descriptors3.csv')


# In[227]:


dfm5=pd.read_csv(r'C:\Users\Ved prakash\descriptors4.csv')


# In[228]:


dfm6=pd.read_csv(r'C:\Users\Ved prakash\descriptors5.csv')


# In[144]:


framen=[dfm1,dfm2,dfm3,dfm4,dfm5,dfm6]


# In[145]:


result4=pd.concat(framen)


# In[146]:


result4


# In[155]:


descriptors6 = from_smiles(['Nc1cccc(S(=O)(=O)N2CCN(S(=O)(=O)c3c(F)cccc3F)CC2)c1',
 'O=S(=O)(NCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1cccc(F)c1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(NC1CCN(S(=O)(=O)c2c(F)cccc2F)CC1)c1ccc2c(c1)OCCO2',
 'CN1CCOc2ccc(S(=O)(=O)N3CCN(S(=O)(=O)c4c(F)cccc4F)CC3)cc21',
 'Cn1ccc2cc(S(=O)(=O)N3CCN(S(=O)(=O)c4c(F)cccc4F)CC3)ccc21',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3)c(=O)c1n2C',
 'CC(C)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'CC(=O)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'Cn1c2cc(S(C)(=O)=O)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 ])

_= from_smiles(['Nc1cccc(S(=O)(=O)N2CCN(S(=O)(=O)c3c(F)cccc3F)CC2)c1',
 'O=S(=O)(NCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1cccc(F)c1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(NC1CCN(S(=O)(=O)c2c(F)cccc2F)CC1)c1ccc2c(c1)OCCO2',
 'CN1CCOc2ccc(S(=O)(=O)N3CCN(S(=O)(=O)c4c(F)cccc4F)CC3)cc21',
 'Cn1ccc2cc(S(=O)(=O)N3CCN(S(=O)(=O)c4c(F)cccc4F)CC3)ccc21',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3)c(=O)c1n2C',
 'CC(C)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'CC(=O)c1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'Cn1c2cc(S(C)(=O)=O)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'Cn1c2cc([S+](C)[O-])sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 ],output_csv='descriptors6.csv')
                          


# In[157]:


import os.path,time
print("last modified:%s"%time.ctime(os.path.getmtime("descriptors6.csv")))
print("created:%s"%time.ctime(os.path.getctime("descriptors6.csv")))


# In[159]:


descriptors7 = from_smiles(['COc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(NC1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)C1)c1c(F)cccc1F',
 'O=S(=O)(NCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'O=S(=O)(NCCCCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'C[C@@H]1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)CCN1S(=O)(=O)c1c(F)cccc1F',
 'O=S(=O)(c1ccc(Cl)cc1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'CC1(C)CCc2cc(S(=O)(=O)N3CCN(S(=O)(=O)c4c(F)cccc4F)CC3)ccc2O1',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3c(F)cccc3F)CC2)cc1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2cc(F)c(F)cc2F)CC1'])
_= from_smiles(['COc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'O=S(=O)(NC1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)C1)c1c(F)cccc1F',
 'O=S(=O)(NCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'O=S(=O)(NCCCCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'C[C@@H]1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)CCN1S(=O)(=O)c1c(F)cccc1F',
 'O=S(=O)(c1ccc(Cl)cc1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1',
 'CC1(C)CCc2cc(S(=O)(=O)N3CCN(S(=O)(=O)c4c(F)cccc4F)CC3)ccc2O1',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3c(F)cccc3F)CC2)cc1',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2cc(F)c(F)cc2F)CC1'],output_csv='descriptors7.csv')


# In[178]:


descriptors8 = from_smiles(['O=S(=O)(c1c(F)cccc1F)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3Cl)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C(C)C',
 'Cn1c2cc([N+](=O)[O-])sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2nccc(Cl)c12'])
_=from_smiles(['O=S(=O)(c1c(F)cccc1F)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3Cl)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C(C)C',
 'Cn1c2cc([N+](=O)[O-])sc2c2cnn(Cc3ccccc3F)c(=O)c21',
 'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2nccc(Cl)c12'],output_csv='descriptors8.csv')



# In[179]:


descriptors8_ = from_smiles([
 'Cc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)c(F)c1',
 'COc1cc(F)c(S(=O)(=O)N2CCN(S(=O)(=O)c3cccc(N)c3)CC2)c(F)c1',
 'O=S(=O)(NCC1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)C1)c1c(F)cccc1F',
 'O=S(=O)(c1ccc(F)cc1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1'])


_=from_smiles(['Cc1ccc(Cn2ncc3c4sc(C)cc4n(C)c3c2=O)c(F)c1',
 'COc1cc(F)c(S(=O)(=O)N2CCN(S(=O)(=O)c3cccc(N)c3)CC2)c(F)c1',
 'O=S(=O)(NCC1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)C1)c1c(F)cccc1F',
 'O=S(=O)(c1ccc(F)cc1)N1CCN(S(=O)(=O)c2ccc3c(c2)OCCO3)CC1'],output_csv='descriptors8_.csv')


# In[180]:




 descriptors9 = from_smiles(['O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2ccccc2F)CC1',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc(OC)cc3)CC2)cc1',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(C(F)(F)F)cc3F)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(C)c3F)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3c(F)c(F)cc(F)c3F)c(=O)c1n2C',
 'C[C@@H]1CN(S(=O)(=O)c2c(F)cccc2F)CCN1S(=O)(=O)c1ccc2c(c1)OCCO2',
 'O=S(=O)(N[C@H]1CC[C@H](NS(=O)(=O)c2c(F)cccc2F)CC1)c1ccc2c(c1)OCCO2',
 'COc1cccc(Cn2ncc3c4sc([S+](C)[O-])cc4n(C)c3c2=O)c1',
 'Cc1cc2c(s1)c1cnn(Cc3c(F)cccc3Cl)c(=O)c1n2C',
 'O=C(c1cc(C2CC2)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1'])
_=from_smiles(['O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2ccccc2F)CC1',
 'COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc(OC)cc3)CC2)cc1',
 'Cc1cc2c(s1)c1cnn(Cc3ccc(C(F)(F)F)cc3F)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3cccc(C)c3F)c(=O)c1n2C',
 'Cc1cc2c(s1)c1cnn(Cc3c(F)c(F)cc(F)c3F)c(=O)c1n2C',
 'C[C@@H]1CN(S(=O)(=O)c2c(F)cccc2F)CCN1S(=O)(=O)c1ccc2c(c1)OCCO2',
 'O=S(=O)(N[C@H]1CC[C@H](NS(=O)(=O)c2c(F)cccc2F)CC1)c1ccc2c(c1)OCCO2',
 'COc1cccc(Cn2ncc3c4sc([S+](C)[O-])cc4n(C)c3c2=O)c1',
 'Cc1cc2c(s1)c1cnn(Cc3c(F)cccc3Cl)c(=O)c1n2C',
 'O=C(c1cc(C2CC2)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1'],output_csv='descriptors9.csv')


# In[181]:



 descriptors10 = from_smiles(['O=C(c1cc(C(F)(F)F)n[nH]1)N(Cc1cccc2c1CCN2)CC1CCC1',
 'O=C(c1cc(Cl)n[nH]1)N(Cc1cccc2[nH]ncc12)CC1CCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccc(F)cc1)CC1CCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1O)CC1CCC1',
 'O=c1cc(Cn2cnc3ccccc32)nc2sccn12',
 'Cc1cccc2nc(Cn3cnc4ccccc43)cc(=O)n12',
 'Cc1ccc2nc(Cn3cnc4ccccc43)cc(=O)n2c1',
 'O=S(=O)(NCCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1ccc2c(c1)OCCO2)C1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'O=S(=O)(N[C@H]1CC[C@@H](NS(=O)(=O)c2c(F)cccc2F)CC1)c1ccc2c(c1)OCCO2',
 'O=C1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)CCN1S(=O)(=O)c1c(F)cccc1F',
 'Cn1c2ccsc2c2ncn(Cc3ccccc3F)c(=O)c21',
 'Cc1cc2c(s1)c1c(C)nn(Cc3ccccc3F)c(=O)c1n2C',
 'Cc1cc2[nH]c3c(=O)n(Cc4ccccc4F)ncc3c2s1',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccc(F)cc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1-c1ccccc1'])
_=from_smiles(['O=C(c1cc(C(F)(F)F)n[nH]1)N(Cc1cccc2c1CCN2)CC1CCC1',
 'O=C(c1cc(Cl)n[nH]1)N(Cc1cccc2[nH]ncc12)CC1CCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccc(F)cc1)CC1CCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1O)CC1CCC1',
 'O=c1cc(Cn2cnc3ccccc32)nc2sccn12',
 'Cc1cccc2nc(Cn3cnc4ccccc43)cc(=O)n12',
 'Cc1ccc2nc(Cn3cnc4ccccc43)cc(=O)n2c1',
 'O=S(=O)(NCCCCNS(=O)(=O)c1c(F)cccc1F)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1ccc2c(c1)OCCO2)C1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'O=S(=O)(N[C@H]1CC[C@@H](NS(=O)(=O)c2c(F)cccc2F)CC1)c1ccc2c(c1)OCCO2',
 'O=C1CN(S(=O)(=O)c2ccc3c(c2)OCCO3)CCN1S(=O)(=O)c1c(F)cccc1F',
 'Cn1c2ccsc2c2ncn(Cc3ccccc3F)c(=O)c21',
 'Cc1cc2c(s1)c1c(C)nn(Cc3ccccc3F)c(=O)c1n2C',
 'Cc1cc2[nH]c3c(=O)n(Cc4ccccc4F)ncc3c2s1',
 'Cc1cc2c(s1)c1cnn(Cc3ccccc3F)c(=O)c1n2C',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccc(F)cc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1-c1ccccc1'],output_csv='descriptors10.csv')
            


# In[ ]:


descriptors11 = from_smiles(['O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1cccs1',
'Cc1cc2c(s1)c1cnn(Cc3c(F)cccc3F)c(=O)c1n2C',
'Cn1c2cc(C=O)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
'Cn1c2ccsc2c2cnn(Cc3ccccc3F)c(=O)c21',
'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1)CC1CC1',
'O=C(c1cc(Br)c[nH]1)N(Cc1ccc(O)cc1)CC1CCC1',
'N#Cc1cc(C(=O)N(Cc2ccc(O)cc2F)CC2CCC2)[nH]n1',
'COc1cccc(NS(=O)(=O)c2cc3c(cc2Cl)NC(=O)CO3)c1',
'Cc1ccc2nc(Cn3c(Cc4ccccc4)nc4ccccc43)cc(=O)n2c1',
'COc1ccccc1N1CCN(C(=O)c2ccc(NS(=O)(=O)c3cccc4cccnc34)cc2)CC1',
'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2nccc(Br)c12',
'Cc1ccc2c(C(=O)CCSC(=S)NCc3cccnc3)c[nH]c2n1',
'O=C(CCSC(=S)NCc1cccnc1)c1cnn2ccccc12',
'O=C(CCSC(=S)NCc1cccnc1)c1cn(-c2ccccc2)c2ncccc12',
'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2ncc(Cl)cc12',
'COc1cccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)c1',
'O=c1ccc2cc(S(=O)(=O)Nc3ccc(C4(O)CSC(=S)N4Cc4cccnc4)cc3)ccc2o1',
'COc1ccccc1N1CCN(C(=O)c2ccc(NS(=O)(=O)c3cccc4cccnc34)cc2)CC1',
'CS(=O)(=O)N1CCc2cc(S(=O)(=O)c3ccc4c(c3)OCCO4)ccc21',
'O=C(COP(=O)(O)O)[C@@H](O)[C@H](O)[C@H](O)COP(=O)(O)O'])
_=from_smiles(['O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1cccs1',
'Cc1cc2c(s1)c1cnn(Cc3c(F)cccc3F)c(=O)c1n2C',
'Cn1c2cc(C=O)sc2c2cnn(Cc3ccccc3F)c(=O)c21',
'Cn1c2ccsc2c2cnn(Cc3ccccc3F)c(=O)c21',
'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1)CC1CC1',
'O=C(c1cc(Br)c[nH]1)N(Cc1ccc(O)cc1)CC1CCC1',
'N#Cc1cc(C(=O)N(Cc2ccc(O)cc2F)CC2CCC2)[nH]n1',
'COc1cccc(NS(=O)(=O)c2cc3c(cc2Cl)NC(=O)CO3)c1',
'Cc1ccc2nc(Cn3c(Cc4ccccc4)nc4ccccc43)cc(=O)n2c1',
'COc1ccccc1N1CCN(C(=O)c2ccc(NS(=O)(=O)c3cccc4cccnc34)cc2)CC1',
'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2nccc(Br)c12',
'Cc1ccc2c(C(=O)CCSC(=S)NCc3cccnc3)c[nH]c2n1',
'O=C(CCSC(=S)NCc1cccnc1)c1cnn2ccccc12',
'O=C(CCSC(=S)NCc1cccnc1)c1cn(-c2ccccc2)c2ncccc12',
'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2ncc(Cl)cc12',
'COc1cccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)c1',
'O=c1ccc2cc(S(=O)(=O)Nc3ccc(C4(O)CSC(=S)N4Cc4cccnc4)cc3)ccc2o1',
'COc1ccccc1N1CCN(C(=O)c2ccc(NS(=O)(=O)c3cccc4cccnc34)cc2)CC1',
'CS(=O)(=O)N1CCc2cc(S(=O)(=O)c3ccc4c(c3)OCCO4)ccc21',
'O=C(COP(=O)(O)O)[C@@H](O)[C@H](O)[C@H](O)COP(=O)(O)O'],output_csv='descriptors11.csv')
   


# In[ ]:



 descriptors12 = from_smiles([ 'O=C(c1cc(Br)c[nH]1)N(Cc1ccc(Cl)cc1)CC1CCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1)CC1CCC1',
 'C=C1C(=O)O[C@H]2[C@H]1CC/C(C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOC(=O)/C1=C/CC[C@@]3(C)O[C@H]3[C@H]3OC(=O)C(=C)[C@@H]3CC1)=C\\CC[C@@]1(C)O[C@@H]21',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2CCc2cccnc2)cc1)c1ccc2c(c1)OCCO2',
 'Cc1cn2c(=O)cc(Cn3cnc4ccccc43)nc2s1',
 'Cc1cccn2c(=O)cc(Cn3cnc4ccccc43)nc12',
 'O=c1cc(Cn2cnc3ccccc32)nc2ccc(Cl)cn12',
 'Cc1ccn2c(=O)cc(Cn3cnc4ccccc43)nc2c1',
 'O=C(c1cc(Cl)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'O=C(c1cc(Br)n[nH]1)N(Cc1cccc2[nH]ncc12)CC1CCC1',
 'O=C(c1cc(Br)n[nH]1)N(Cc1cccc2[nH]c(=O)sc12)CC1CCC1',
 'N#Cc1ccc(CN(CC2CCC2)C(=O)c2cc(Br)c[nH]2)cc1',
 'Nc1ccc(CN(CC2CCC2)C(=O)c2cc(Br)c[nH]2)cc1',
 'CCCN(Cc1ccccc1)C(=O)c1cc(Br)c[nH]1',
 'O=C(c1cc(C(F)(F)F)n[nH]1)N(Cc1cccc2[nH]ccc12)CC1CCC1',
 'O=C(c1cc(C(F)(F)F)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'O=C(c1cc(Br)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'COc1ccc(CN(CC2CCC2)C(=O)c2cc(Br)c[nH]2)cc1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'CCCCCN(Cc1ccccc1)C(=O)c1cc(Br)c[nH]1'])
_=from_smiles(['O=C(c1cc(Br)c[nH]1)N(Cc1ccc(Cl)cc1)CC1CCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1)CC1CCC1',
 'C=C1C(=O)O[C@H]2[C@H]1CC/C(C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOC(=O)/C1=C/CC[C@@]3(C)O[C@H]3[C@H]3OC(=O)C(=C)[C@@H]3CC1)=C\\CC[C@@]1(C)O[C@@H]21',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2CCc2cccnc2)cc1)c1ccc2c(c1)OCCO2',
 'Cc1cn2c(=O)cc(Cn3cnc4ccccc43)nc2s1',
 'Cc1cccn2c(=O)cc(Cn3cnc4ccccc43)nc12',
 'O=c1cc(Cn2cnc3ccccc32)nc2ccc(Cl)cn12',
 'Cc1ccn2c(=O)cc(Cn3cnc4ccccc43)nc2c1',
 'O=C(c1cc(Cl)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'O=C(c1cc(Br)n[nH]1)N(Cc1cccc2[nH]ncc12)CC1CCC1',
 'O=C(c1cc(Br)n[nH]1)N(Cc1cccc2[nH]c(=O)sc12)CC1CCC1',
 'N#Cc1ccc(CN(CC2CCC2)C(=O)c2cc(Br)c[nH]2)cc1',
 'Nc1ccc(CN(CC2CCC2)C(=O)c2cc(Br)c[nH]2)cc1',
 'CCCN(Cc1ccccc1)C(=O)c1cc(Br)c[nH]1',
 'O=C(c1cc(C(F)(F)F)n[nH]1)N(Cc1cccc2[nH]ccc12)CC1CCC1',
 'O=C(c1cc(C(F)(F)F)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'O=C(c1cc(Br)n[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'COc1ccc(CN(CC2CCC2)C(=O)c2cc(Br)c[nH]2)cc1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccc(O)cc1F)CC1CCC1',
 'CCCCCN(Cc1ccccc1)C(=O)c1cc(Br)c[nH]1'],output_csv='descriptors12.csv')


# In[214]:



 descriptors13 = from_smiles(['O=c1cc(Cn2cnc3ccccc32)nc2ccccn12',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1)CC1CCCCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1cccc(F)c1)CC1CCC1',
 'CCc1nn2c(=O)cc(Cn3cnc4ccccc43)nc2s1',
 'CCc1ccc(C)n2c(=O)cc(Cn3cnc4ccccc43)nc12',
 'O=C(c1cc(Br)c[nH]1)N(Cc1cccc(O)c1)CC1CCC1',
 'COc1ccc(Cl)cc1S(=O)(=O)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1F',
 'Cn1cc(C(=O)c2cccc(Cl)c2Cl)cc1C(=O)NCCNC(=O)c1cc(C(=O)c2cccc(Cl)c2Cl)cn1C',
 'Cc1cc(C)c(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)c(C)c1',
 'O=[N+]([O-])c1ccccc1S(=O)(=O)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1cccc(F)c1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cnccn2)cc1)c1ccc2c(c1)OCCO2',
 'O=c1cc(Cn2cnc3ccccc32)nc2scc(-c3ccccc3)n12',
 'Cn1cc(C(=O)CCSC(=S)NCc2cccnc2)c2cccnc21',
 'O=C(CCSC(=S)NCc1cccnc1)c1cn(C2CC2)c2ncccc12',
 'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2ncc(-c3ccc(Cl)cc3)cc12',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cnc3ccccc3c2)cc1)c1ccc2c(c1)OCCO2',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2ccc3nccn3c2)cc1)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'Cn1cc(C(=O)c2cccc(Cl)c2Cl)cc1C(N)=O',
 'COc1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'O=C1CCc2cc(S(=O)(=O)Nc3ccc(C4(O)CSC(=S)N4Cc4cccnc4)cc3)ccc2N1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1N1CCOCC1'])
_=from_smiles(['O=c1cc(Cn2cnc3ccccc32)nc2ccccn12',
 'O=C(c1cc(Br)c[nH]1)N(Cc1ccccc1)CC1CCCCC1',
 'O=C(c1cc(Br)c[nH]1)N(Cc1cccc(F)c1)CC1CCC1',
 'CCc1nn2c(=O)cc(Cn3cnc4ccccc43)nc2s1',
 'CCc1ccc(C)n2c(=O)cc(Cn3cnc4ccccc43)nc12',
 'O=C(c1cc(Br)c[nH]1)N(Cc1cccc(O)c1)CC1CCC1',
 'COc1ccc(Cl)cc1S(=O)(=O)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1F',
 'Cn1cc(C(=O)c2cccc(Cl)c2Cl)cc1C(=O)NCCNC(=O)c1cc(C(=O)c2cccc(Cl)c2Cl)cn1C',
 'Cc1cc(C)c(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)c(C)c1',
 'O=[N+]([O-])c1ccccc1S(=O)(=O)Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1',
 'O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1cccc(F)c1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cnccn2)cc1)c1ccc2c(c1)OCCO2',
 'O=c1cc(Cn2cnc3ccccc32)nc2scc(-c3ccccc3)n12',
 'Cn1cc(C(=O)CCSC(=S)NCc2cccnc2)c2cccnc21',
 'O=C(CCSC(=S)NCc1cccnc1)c1cn(C2CC2)c2ncccc12',
 'O=C(CCSC(=S)NCc1cccnc1)c1c[nH]c2ncc(-c3ccc(Cl)cc3)cc12',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cnc3ccccc3c2)cc1)c1ccc2c(c1)OCCO2',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2ccc3nccn3c2)cc1)c1ccc2c(c1)OCCO2',
 'O=S(=O)(c1ccc2c(c1)OCCO2)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1',
 'Cn1cc(C(=O)c2cccc(Cl)c2Cl)cc1C(N)=O',
 'COc1ccc(S(=O)(=O)Nc2ccc(C3(O)CSC(=S)N3Cc3cccnc3)cc2)cc1',
 'O=C1CCc2cc(S(=O)(=O)Nc3ccc(C4(O)CSC(=S)N4Cc4cccnc4)cc3)ccc2N1',
 'O=S(=O)(Nc1ccc(C2(O)CSC(=S)N2Cc2cccnc2)cc1)c1ccccc1N1CCOCC1'],output_csv='descriptors13.csv')


# In[229]:


dfm7=pd.read_csv(r'C:\Users\Ved prakash\descriptors6.csv')


# In[230]:


dfm8=pd.read_csv(r'C:\Users\Ved prakash\descriptors7.csv')


# In[231]:


dfm9=pd.read_csv(r'C:\Users\Ved prakash\descriptors8.csv')


# In[232]:


dfm10=pd.read_csv(r'C:\Users\Ved prakash\descriptors8_.csv')


# In[233]:


dfm11=pd.read_csv(r'C:\Users\Ved prakash\descriptors9.csv')


# In[234]:


dfm12=pd.read_csv(r'C:\Users\Ved prakash\descriptors10.csv')


# In[235]:


dfm13=pd.read_csv(r'C:\Users\Ved prakash\descriptors11.csv')


# In[236]:


dfm14=pd.read_csv(r'C:\Users\Ved prakash\descriptors12.csv')


# In[237]:


dfm15=pd.read_csv(r'C:\Users\Ved prakash\descriptors13.csv')


# In[238]:


framen1=[dfm1,dfm2,dfm3,dfm4,dfm5,dfm6,dfm7,dfm8,dfm9,dfm10,dfm11,dfm12,dfm13,dfm14,dfm15]


# In[239]:


results=pd.concat(framen1)


# In[240]:


results


# In[ ]:




