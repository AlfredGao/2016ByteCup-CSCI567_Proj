
# coding: utf-8

# In[23]:

import pandas as pd
import numpy as np
import pickle


# In[36]:

def load_weight(path):
    U_Matrix, Q_Matrix, overall_mean, p_U, p_Q = pickle.load(open(path,'r'))
    return U_Matrix, Q_Matrix, overall_mean, p_U, p_Q


# In[39]:

U_Matrix, Q_Matrix, overall_mean, p_U, p_Q = load_weight("k70iter600.model")


# In[45]:

def predict(user, question):
    score = overall_mean + p_U[user] + p_Q[question] + np.dot(U_Matrix[user], Q_Matrix[question].T)
    score = max(score, 0.)
    score = min(score, 1.)
    return float(score)


# In[51]:

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


# In[52]:

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


# In[53]:

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
            score = predict(dict_u[u_id], dict_q[q_id])
        except Exception as e:
            score = 0.
#             print q_id + ',' + u_id + " Unable to predict"

        if score < 1e-5:
            score = 0.

        result = q_id + "," + u_id + "," + str(score) + '\n'
        result_file.write(result)
result_file.close()

print 'load weight and predict Sucessfully!'
