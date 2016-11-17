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
    MF.factorize(k =70, iter= 1, alpha = 0.0018, beta = 0.058, svdpp=False)
    #TODO Implement save_model function
    #MF.save_model()
    end_time = time.clock()

    result_path = '../data/validate/result.txt'
    NN_CF_Feature = '../data/predictTrain/predict_on_train.txt'
    test_submit_path = '../data/submit/test_submit.txt'

    result_file = open(result_path,'w')
    NN_CF_Feature_file = open(NN_CF_Feature,'w')
    test_submit_file = open(test_submit_path,'w')

    test_file = open('../data/test_nolabel.txt','r')
    vali_file = open('../data/validate_nolabel.txt','r')
    train_file = open('../data/invited_info_train.txt','r')

    result_file.write('qid,uid,label\n')
    test_submit_file.write('qid,uid.label\n')
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
    print "Sucessfully create the validate file!"

    test_unpredict_file = open('../data/submit/test_unpredict.txt','w')
    test_unpredict_file.write('This is unpredict data for test file\n')
    test_unpredict_file.write('qid,uuid\n')
    is_head = True
    for fileline in test_file:
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
                unpredict = q_id + ',' + u_id + '\n'
                test_unpredict_file.write(unpredict)

            if score < 1e-5:
                score = 0.

            result = q_id + "," + u_id + "," + str(score) + '\n'
            test_submit_file.write(result)
    test_unpredict_file.close()
    test_submit_file.close()
    print "Sucessfully create test file!"

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
                score = MF.predict(dict_u[u_id], dict_q[q_id])
            except Exception as e:
                score = 0.
                unpredict = q_id + ',' + u_id + '\n'
                print 'Fucking!'

            if score < 1e-5:
                score = 0.

            result = q_id + "," + u_id + "," + str(score) + '\n'
            NN_CF_Feature_file.write(result)
    train_file.close()
    NN_CF_Feature_file.close()
    print "Sucessfully create predict on train file!"

    # try:
    #     MF.save_model(mean_path, p_U_path, p_Q_path, U_M_path, Q_M_path)
    # except Exception as e:
    #     print 'Save model failed'
    MF.save_weight("../savemodel/k70iter600.model")
    print "Suceesfully save model!"
