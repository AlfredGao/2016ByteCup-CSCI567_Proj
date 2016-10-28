from Matrix_Factorize import *

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


if __name__ == "__main__":

    #
    #
    # # print MF.predict(2491,10)
    # #
    # #
    question_hash_path = '../data/invited_question_info_hash.txt'
    user_hash_path = '../data/invited_user_info_hash.txt'

    dict_q = load_hashdata(question_hash_path)
    dict_u = load_hashdata(user_hash_path)

    # print dict_q
    # print dict_u

    MF = MX_F()
    MF.load_data('../data/train_hash_shuff.txt', '\t')
    start_time = time.clock()
    MF.factorize(k = 30, iter=250, alpha = 0.01, beta = 0.05)
    end_time = time.clock()

    result_file = open('../data/result.txt','w')
    vali_file = open('../data/validate_nolabel.txt','r')
    result_file.write('qid,uid,label\n')
    is_head = True
    for fileline in vali_file:
        if is_head:
            is_head = False
            continue
        else:
            fileline = fileline.strip('\n\r').split(',')
            # print fileline
            q_id = fileline[0]
            u_id = fileline[1]
            # print dict_u[u_id]
            # print dict_q[q_id]
            try:
                score = MF.predict(dict_u[u_id], dict_q[q_id])
                # print score
            except Exception as e:
                score = 0.
                print q_id + ',' + u_id + " Unable to predict"

            if score < 1e-5:
                score = 0.

            result = q_id + "," + u_id + "," + str(score) + '\n'
            result_file.write(result)
    result_file.close()
