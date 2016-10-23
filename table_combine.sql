DELETE FROM train_data;
INSERT INTO train_data
SELECT  invited_info_train.qid,
			question_info.qtag, question_info.word_feature as qword, 
			question_info.char_feature as qchar, question_info.num_like, 
			question_info.num_answear, question_info.num_HQanswear,
			invited_info_train.uuid, 
			user_info.word_feature as uword,
			user_info.char_feature as uchar,
			user_info.u_label,
			invited_info_train.is_answear as y_label
from invited_info_train
inner join question_info
	on invited_info_train.qid = question_info.qid
inner join user_info
	on invited_info_train.uuid = user_info.uuid;