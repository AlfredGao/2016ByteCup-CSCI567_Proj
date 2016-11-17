
# coding: utf-8

# In[31]:

import pandas as pd
import numpy as np


# In[32]:

mean = pd.read_csv('overall_mean.model',names=['overall_mean'])
mean


# In[33]:

p_q = pd.read_csv('p_Q.model', names=['penalty_Q'])
p_q


# In[34]:

p_u = pd.read_csv('p_U.model', names=['penalty_U'])
p_u


# In[35]:

q_m = pd.read_csv('Q_Maxtrix.model',names = ['q_m'])
q_m


# In[47]:

u_m = pd.read_csv('U_Maxtrix.model',names = ['u_m'])
float(u_m.loc[3])
np.dot(u_m.loc[3], q_m.loc[5].T) + 1


# In[37]:

def load_hashdata(path):
    d = {}
    try:
        file_question_hash = open(path,'r')
    except Exception as e:
        raise ValueError('Cannot find txt file pls run Map_data.py script first')
    for fileline in file_question_hash:
        fileline = fileline.strip('\r\n').split('\t')
        d[fileline[0]] = int(fileline[1])

    return d


# In[38]:

question_hash_path = '../data/invited_question_info_hash.txt'
user_hash_path = '../data/invited_user_info_hash.txt'


# In[39]:

dict_q = load_hashdata(question_hash_path)
dict_u = load_hashdata(user_hash_path)


# In[48]:

def predict_load(user, question):
    score = float(mean.loc[0]) + float(p_u.loc[user]) + float(p_q.loc[question]) + np.dot(u_m.loc[user], q_m.loc[question].T)
    score = max(score, 0.)
    score = min(score, 1.)
    return float(score)


# In[49]:

result_path = 'result_loadmodel.txt'
    # NN_CF_Feature = '../data/NN_CF_Feature.txt'
# test_submit_path = '../data/submit/test_submit.txt'

result_file = open(result_path,'w')
    # NN_CF_Feature_file = open(NN_CF_Feature,'w')
# test_submit_file = open(test_submit_path,'w')

# test_file = open('../data/test_nolabel.txt','r')
vali_file = open('../data/validate_nolabel.txt','r')
    # train_file = open('../data/invited_info_NN_bj.txt','r')

result_file.write('qid,uid,label\n')
# test_submit_file.write('qid,uid.label\n')
is_head = True
for fileline in vali_file:
    if is_head:
        is_head = False
        continue
    else:
        fileline = fileline.strip('\n\r').split(',')
        q_id = fileline[0]
        u_id = fileline[1]

        try:
            score = predict_load(dict_u[u_id], dict_q[q_id])
        except Exception as e:
            score = 0.
#             print q_id + ',' + u_id + " Unable to predict"

        if score < 1e-5:
            score = 0.

        result = q_id + "," + u_id + "," + str(score) + '\n'
        result_file.write(result)
result_file.close()


# In[ ]:



