import pandas as pd
import numpy as np
from sklearn import linear_model

invited_info = pd.read_csv('invited_info_csv.txt', sep='\t')
question_info = pd.read_csv('question_info_csv.txt', sep='\t')
user_info = pd.read_csv('user_info_csv.txt', sep='\t')

all_info = invited_info.merge(question_info, left_on='qid', right_on='qid', how='left')
all_info = all_info.merge(user_info, left_on='uid', right_on='uid',how='left')
label_series = all_info['label']
label_array = np.array(label_series)
feature_dataframe = all_info.loc[:, 'qwid':'utag_142']
feature_name = list(feature_dataframe.columns)
feature_matrix = feature_dataframe.as_matrix()
validate_nolabel = pd.read_csv('validate_nolabel.txt', sep=',')
validate_set = validate_nolabel.merge(question_info, left_on='qid', right_on='qid',how='left')
validate_set = validate_set.merge(user_info, left_on='uid', right_on='uid', how='left')
validate_set_matrix = validate_set.loc[:,'qwid':'utag_142'].as_matrix()
reg = linear_model.BayesianRidge()
reg.fit(feature_matrix, label_array)
pre_vali_y =reg.predict(validate_set_matrix)
validate_nolabel['label'] = pre_vali_y
validate_nolabel.to_csv('temp.csv', sep=',', index=False)






