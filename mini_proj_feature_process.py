import numpy as np
import pandas as pd
import seaborn as sb
from copy import deepcopy

invited_info_train = pd.read_csv("invited_info_train.txt", sep='\t', names =[
    'qid', 'uid', 'label'])
invited_info_train_copy = deepcopy(invited_info_train)
invited_info_train_copy.loc[invited_info_train_copy.duplicated(['qid','uid'], keep=False), 'label'] = 0.5
invited_info_train_copy = invited_info_train_copy.drop_duplicates(['qid','uid'], keep='first')
question_info = pd.read_csv("question_info.txt", sep='\t', names=['qid','qtag','qwid','qcid','#upvotes','#ans','#tqans'])
question_info_copy = deepcopy(question_info)
qtag_unique = question_info_copy['qtag'].unique()
for i in range(len(qtag_unique)):
    question_info_copy['qtag_{}'.format(i)] = (question_info_copy['qtag'] == i).astype(int)
question_info_copy = question_info_copy.drop(['qtag'], axis=1)
wid_new_id = np.arange(len(question_info_copy['qwid'].unique()))
np.random.shuffle(wid_new_id)
cid_new_id = np.arange(len(question_info_copy['qcid'].unique()))
np.random.shuffle(cid_new_id)

def replace_id(new_id_list, values):
    values = deepcopy(values)
    output_dict = {}
    uniq_values = values.unique()
    for i in range(len(new_id_list)):
        output_dict[uniq_values[i]] = new_id_list[i]
    for i in range(len(values)):
        values[i] = output_dict[values[i]]
    return values

question_info_copy['qwid'] = replace_id(wid_new_id, question_info_copy['qwid'])
question_info_copy['qcid'] = replace_id(cid_new_id, question_info_copy['qcid'])

def normal_col(column):
    mean = column.mean()
    std = column.std()
    column = (column - mean) / std
    return column

question_info_copy['qwid'] = normal_col(question_info_copy['qwid'])
question_info_copy['qcid'] = normal_col(question_info_copy['qcid'])
question_info_copy['#upvotes/#ans'] = question_info_copy['#upvotes'] / question_info_copy['#ans']
question_info_copy['#tqans/#ans'] = question_info_copy['#tqans'] / question_info_copy['#ans']
question_info_copy.loc[question_info_copy['#ans'] == 0, '#upvotes/#ans'] = 0.0
question_info_copy.loc[question_info_copy['#ans'] == 0, '#tqans/#ans'] = 0.0
question_info_copy['#upvotes'] = normal_col(question_info_copy['#upvotes'])
question_info_copy['#ans'] = normal_col(question_info_copy['#ans'])
question_info_copy['#tqans'] = normal_col(question_info_copy['#tqans'])
question_info_copy['#upvotes/#ans'] = normal_col(question_info_copy['#upvotes/#ans'])
question_info_copy['#tqans/#ans'] = normal_col(question_info_copy['#tqans/#ans'])

user_info = pd.read_csv("user_info.txt", sep='\t', names=['uid', 'utag', 'uwid', 'ucid'])
user_info_copy = deepcopy(user_info)
uwid_new_id = np.arange(len(user_info_copy['uwid'].unique()))
np.random.shuffle(uwid_new_id)
ucid_new_id = np.arange(len(user_info_copy['ucid'].unique()))
np.random.shuffle(ucid_new_id)
user_info_copy['uwid'] = replace_id(uwid_new_id, user_info_copy['uwid'])
user_info_copy['ucid'] = replace_id(ucid_new_id, user_info_copy['ucid'])
for i in range(len(user_info_copy['utag'])):
    user_info_copy['utag'][i] = map(int, user_info_copy['utag'][i].split('/'))
max_utag = max(user_info_copy['utag'][0])
for i in range(len(user_info_copy['utag'])):
    if max_utag < max(user_info_copy['utag'][i]):
        max_utag = max(user_info_copy['utag'][i])
for i in range(max_utag+1):
    user_info_copy['utag_{0}'.format(i)] = 0
for i in range(len(user_info_copy)):
    num_list = user_info_copy.loc[i, 'utag']
    for num in num_list:
        user_info_copy.loc[i, 'utag_{0}'.format(num)] = 1
user_info_copy = user_info_copy.drop(['utag'], axis=1)
user_info_copy['uwid'] = normal_col(user_info_copy['uwid'])
user_info_copy['ucid'] = normal_col(user_info_copy['ucid'])
invited_info_train_copy.to_csv('invited_info_csv.txt', sep='\t', index=False)
question_info_copy.to_csv('question_info_csv.txt', sep='\t', index=False)
user_info_copy.to_csv('user_info_csv.txt', sep='\t', index=False)