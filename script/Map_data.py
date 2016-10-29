import numpy as np


def get_dict():
    dict_u = {}
    dict_q = {}
    try:
        file_train = open('../data/invited_info_train.txt','r')
    except Exception as e:
        print "Cannot find the file of training data, pls check the path"
        return dict_u,dict_q
    u_index = 0
    q_index = 0
    is_head = True
    for fileline in file_train:
        if is_head:
            is_head = False
            continue
        else:
            fileline = fileline.strip('\n\r').split('\t')
            # print fileline
            try:
                dict_u[fileline[1]]
            except Exception as e:
                temp = {fileline[1]: u_index}
                dict_u.update(temp)
                u_index = u_index + 1
            try:
                dict_q[fileline[0]]
            except Exception as e:
                temp = {fileline[0]: q_index}
                dict_q.update(temp)
                q_index = q_index + 1
    file_train.close()
    return dict_u, dict_q


def create_hashtrain(dict_u, dict_q, question_hash, user_hash):
    file_hash = open('../data/invited_info_train_hash.txt','w')
    file_question = open(question_hash,'w')
    file_user = open(user_hash, 'w')
    file_train = open('../data/invited_info_train.txt','r')
    is_head = True;
    for fileline in file_train:
        if is_head:
            hash_line = 'qid' + '\t' + 'uuid' + '\t' + 'is_answear' + '\n'
            is_head = False
        else:
            fileline = fileline.strip('\n\r').split('\t')
            hash_line = str(dict_q[fileline[0]]) + '\t' + str(dict_u[fileline[1]]) + '\t' + str(fileline[2]) +  '\n'
        file_hash.write(hash_line)

    file_hash.close()
    file_train.close()


    for k, v in dict_u.iteritems():
        line = str(k) + '\t' + str(v) + '\n'
        file_user.write(line)

    for k, v in dict_q.iteritems():
        line = str(k) + '\t' + str(v) + '\n'
        file_question.write(line)
    file_user.close()
    file_question.close()


# def create_hashdata(path, dict_type, filename):
#     try:
#         file_origin = open(path,'r')
#     except Exception as e:
#         print "Cannot find the file, pls check your file path"
#     file_hash = open(filename, 'w')
#     for line in file_origin:
#         line = line.strip('\r\n').split('\t')
#         hash_line = str(line[0]) + '\t' + str(dict_type[line[0]]) + '\n'
#         file_hash.write(hash_line)
#     file_origin.close()
#     file_hash.close()


def hash_data():
    dict_u, dict_q = get_dict()
    """
    create the path for file
    """
    
    # question_path = '../data/question_info.txt'
    # user_path = '../data/user_info.txt'
    question_hash = '../data/invited_question_info_hash.txt'
    user_hash = '../data/invited_user_info_hash.txt'
    create_hashtrain(dict_u, dict_q, question_hash, user_hash)
    # """
    # create hash file
    # """
    # create_hashdata(question_path, dict_q, question_hash)
    # create_hashdata(user_path, dict_u, user_hash)


if __name__ == '__main__':
    hash_data()
    print "Map of data has been completed!"
