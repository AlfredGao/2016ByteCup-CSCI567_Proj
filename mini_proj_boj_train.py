
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

invited_info = pd.read_csv('invited_info_csv.txt', sep='\t')
question_info = pd.read_csv('question_info_csv.txt', sep='\t')
user_info = pd.read_csv('user_info_csv.txt', sep='\t')


# In[3]:

invited_info = invited_info.drop(['Unnamed: 0'], axis=1)
question_info = question_info.drop(['Unnamed: 0'], axis=1)
user_info = user_info.drop(['Unnamed: 0'], axis=1)


# In[4]:

all_info = invited_info.merge(question_info, left_on='qid', right_on='qid', how='left')


# In[5]:

all_info = all_info.merge(user_info, left_on='uid', right_on='uid', how='left')


# In[6]:

# all_info.head()


# In[7]:

label_series = all_info['label']


# In[8]:

label_array = np.array(label_series)


# In[9]:

feature_dataframe = all_info.loc[:, 'qwid':'utag_142']


# In[76]:

feature_name = list(feature_dataframe.columns)
# feature_name


# In[10]:

feature_matrix = feature_dataframe.as_matrix()


# In[11]:

# from sklearn.feature_selection import RFE
# from sklearn.linear_model import SGDRegressor
# model = SGDRegressor()
# selector = RFE(model, step=1)
# selector = selector.fit(feature_matrix, label_array)

# model.fit(feature_matrix, label_array)


# In[12]:

validate_nolabel = pd.read_csv('./bytecup2016data/validate_nolabel.txt', sep=',')


# In[13]:

# validate_nolabel.head()


# In[14]:

valitate_set = validate_nolabel.merge(question_info, left_on='qid', right_on='qid', how='left')


# In[15]:

valitate_set = valitate_set.merge(user_info, left_on='uid', right_on='uid', how='left')


# In[16]:

# valitate_set.head()


# In[17]:

valitate_set_matrix = valitate_set.loc[:,'qwid':'utag_142'].as_matrix()


# In[18]:

valitate_set_matrix.shape


# In[19]:

# predict_result = model.predict(valitate_set_matrix)
# predict_result

# validate_nolabel['label'] = predict_result + 0.2


# In[20]:

# validate_nolabel.to_csv('validate_nolabel.csv', sep=',', index=False)


# In[54]:

from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=90)
ipca.fit(feature_matrix)
feature_matrix_pca = ipca.transform(feature_matrix) 


# In[55]:

# feature_matrix_pca.shape


# In[56]:

ipca2 = IncrementalPCA(n_components=90)
ipca2.fit(valitate_set_matrix)
valitate_set_matrix_pca = ipca2.transform(valitate_set_matrix)


# In[57]:

# valitate_set_matrix_pca.shape


# In[25]:

# from sklearn.linear_model import SGDRegressor
# model = SGDRegressor()
# model.fit(feature_matrix_pca, label_array)


# In[26]:

# predict_result = model.predict(valitate_set_matrix_pca)


# In[27]:

# max(predict_result)


# In[28]:

import xgboost as xgb


# In[78]:

dtrain = xgb.DMatrix(feature_matrix[:160608], label=label_array[:160608], feature_names=feature_name)


# In[79]:

dtest = xgb.DMatrix(feature_matrix[160608:], label=label_array[160608:], feature_names=feature_name)


# In[80]:

# dtrain


# In[81]:

param = {'bst:max_depth':6, 'bst:eta':1, 'silent':1, 'objective':'reg:linear' }
param['nthread'] = 4
param['eval_metric'] = 'auc'


# In[82]:

num_round = 30


# In[83]:

evallist  = [(dtest,'eval'), (dtrain,'train')]


# In[84]:

bst = xgb.train( param.items(), dtrain, num_round, evallist)


# In[85]:

# %matplotlib inline
xgb.plot_importance(bst, height=0.001)


# In[89]:

# bst.get_fscore()


# In[93]:

dvalidate = xgb.DMatrix(valitate_set_matrix, feature_names=feature_name)


# In[94]:

pre = bst.predict(dvalidate)
from graphviz import *


# In[106]:

ax1 = xgb.plot_tree(bst)


# In[102]:




# In[40]:

validate_nolabel['label'] = pre


# In[41]:

validate_nolabel.to_csv('validate_nolabel.csv', sep=',', index=False)


# In[ ]:
plt.show()



