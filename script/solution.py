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
    MF.factorize(k = 2, iter= 1, alpha = 0.0015, beta = 0.05)
    #TODO Implement save_model function
    #MF.save_model()
    end_time = time.clock()

    result_path = '../data/result_'  + '.txt'
    NN_CF_Feature = '../data/NN_CF_Feature.txt'

    result_file = open(result_path,'w')
    NN_CF_Feature_file = open(NN_CF_Feature,'w')
    vali_file = open('../data/validate_nolabel.txt','r')
    train_file = open('../data/invited_info_NN_bj.txt','r')

    result_file.write('qid,uid,label\n')
    NN_CF_Feature_file.write('qid,uid,feature_CF\n')
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
                score = MF.predict(dict_u[u_id], dict_q[q_id])
            except Exception as e:
                score = 0.
                print q_id + ',' + u_id + " Unable to predict"

            if score < 1e-5:
                score = 0.

            result = q_id + "," + u_id + "," + str(score) + '\n'
            result_file.write(result)
    result_file.close()

    print "Constructing NN Feature....... :)"
    is_head = True
    for fileline in train_file:
        if is_head:
            is_head = False
            continue
        else:
            fileline = fileline.strip('\n\r').split('\t')
            q_id = fileline[0]
            u_id = fileline[1]


            try:
                score = MF.predict(dict_u[u_id],dict_q[q_id])
            except Exception as e:
                score = 0.
                print q_id + ',' + u_id + " Unable to predict"

            if score < 1e-5:
                score = 0.

            result = q_id + "," + u_id + "," + str(score) + '\n'
            NN_CF_Feature_file.write(result)
    NN_CF_Feature_file.close()
