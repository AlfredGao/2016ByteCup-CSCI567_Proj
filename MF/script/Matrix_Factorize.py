import numpy as np
import sys
import pandas as pd
import scipy
from scipy import sparse
import time

class MX_F():
    def __init__(self):
        self._u_num = -1
        self._q_num = -1
        self._f_num = -1
        # self._data = Data()


    def load_data(self, path, sep):
        dict_type = {'qid':int, 'uuid':int, 'is_answear':float}
        # print path
        # print sep
        train = pd.read_csv(path, sep = sep,dtype = dict_type)
        self.pd_train = train
        # print train
        self.np_train = np.array(train)
        # print self.np_train
        # self._data.load(path, sep=sep, format={'col':0,'row':1, 'value':2, 'ids':'int'})


    def _init_U_Matrix(self):
        """
        Init the user matrix for matrix factorize
        """

        if self._u_num == -1 or self._f_num == -1:
            raise ValueError('Error: Miss the num of user and num of feature!')
        else:
            self.U_Maxtrix = 0.015*np.random.random((self._u_num, self._f_num))/np.sqrt(self._f_num)
            print self.U_Maxtrix.shape


    def _init_Q_Matrix(self):
        """
        Init the question matrix for matrix facorize
        """

        if self._q_num == -1 or self._f_num == -1:
            raise ValueError('Error: Miss the num of question and num of feature!')
        else:
            self.Q_Maxtrix = 0.015*np.random.random((self._q_num, self._f_num,))/np.sqrt(self._f_num)
            print self.Q_Maxtrix.shape

    def _init_Y_Matrix(self):

        if self._q_num == -1 or self._f_num == -1:
            raise ValueError('What fuck are you doing!')
        else:
            self.Y_Matrix = 0.015*np.random.random((self._q_num, self._f_num))/np.sqrt(self._f_num)


    def gradient_descent(self, alpha, beta, svdpp = False):
        """
        Apply gradient descent to matrix factorize
        """
        U_M = self.U_Maxtrix
        Q_M = self.Q_Maxtrix
        penalty_U = self.p_U
        penalty_Q = self.p_Q

        mask = self.mask
        mse = (self.data - np.dot(U_M, Q_M.T) - self.overall_mean - penalty_U.repeat(self._q_num, axis = 1) - penalty_Q.repeat(self._u_num, axis = 1).T)
        if svdpp:
            Y_M = self.Y_Matrix
            Ru = self.Ru
            num_answer = (np.dot((mask*mse),Q_M) * Ru.repeat(self._f_num, axis=1))
            self.U_Maxtrix = U_M + alpha*(np.dot(mask*mse, Q_M) - beta*U_M)
            self.Q_Maxtrix = Q_M + alpha*( np.dot((mask*mse).T,U_M + np.dot(self.data,Y_M)*(Ru.repeat(self._f_num,axis=1)) ) - beta*Q_M)
            for i in range(self._u_num):
                temp = np.dot(self.data[i].reshape(-1,1),num_answer[i].reshape(1,-1))
                self.Y_M = Y_M + alpha *( temp - beta*Y_M )
            self.p_U = penalty_U + alpha*((mask*mse).sum(axis=1).reshape(penalty_U.shape) - beta*penalty_U)
            self.p_Q = penalty_Q + alpha*((mask*mse).sum(axis=0).reshape(penalty_Q.shape) - beta*penalty_Q)
        else:
            self.U_Maxtrix = U_M + alpha*(np.dot(mask*mse, Q_M) - beta*U_M)
            self.Q_Maxtrix = Q_M + alpha*(np.dot((mask*mse).T, U_M) - beta*Q_M)
            self.p_U = penalty_U + alpha*((mask*mse).sum(axis=1).reshape(penalty_U.shape) - beta*penalty_U)
            self.p_Q = penalty_Q + alpha*((mask*mse).sum(axis=0).reshape(penalty_Q.shape) - beta*penalty_Q)
        return ((mask*mse)**2).sum()



    def factorize(self, iter, k, alpha, beta, svdpp=False):
        self._u_num = self.pd_train['uuid'].max() + 1
        self._q_num = self.pd_train['qid'].max() + 1
        self._f_num = k

        self._init_Q_Matrix()
        self._init_U_Matrix()
        self._init_Y_Matrix()
        # print 'row max is',self._data.row_max + 1
        # print 'row max is', self.pd_train['uuid'].max() + 1
        # print 'col max is',self._data.col_max + 1
        # print 'col max is', self.pd_train['qid'].max() + 1
        #TODO what is fuck of following??
        # mask_test =  scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( (np.ones(len(self.pd_train)) ,( np.array(self.pd_train['uuid']),np.array(self.pd_train['qid']) ) ) ) ).toarray()
        self.data = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( (np.array(self.pd_train['is_answear']),(np.array(self.pd_train['uuid']),np.array(self.pd_train['qid'])) ) )).toarray()
        # print 'I am not killed before data'
        # print self.data
        # self.data = self._data.get_in_numpy_format()
        # print self.data
        # test_data = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( np.array(self.pd_train['is_answear']),(np.array(self.pd_train['uuid']),np.array(self.pd_train['qid'])) ) )
        self.mask = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( (np.ones(len(self.pd_train)) ,( np.array(self.pd_train['uuid']),np.array(self.pd_train['qid']) ) ) ) ).toarray()
        # print 'I am not killed before mask'
        # self.mask = self._data.get_mask()
        # mask_test = self._data.get_mask()
        # print 'mask test', mask_test.shape
        # print 'not test',self.mask.sum()
        # print 'test',mask_test.sum()
        self.overall_mean = self.data.sum()/self.mask.sum()
        self.p_U = np.zeros((self._u_num, 1))
        self.p_Q = np.zeros((self._q_num, 1))
        self.Ru = (( self.data.sum(axis=1) )** 0.5).reshape(-1,1)

        # cost = self.gradient_descent(alpha, beta)
        # print 'I am not killed before loop'
        for i in range(iter):
            start_time = time.clock()
            cost = self.gradient_descent(alpha, beta, svdpp)
            end_time = time.clock()
            print 'Iteration:' + str(i + 1) + ': cost is ' + str(cost)
            print 'Time of Iteration is: ' + str(end_time - start_time)


    def save_model(self):
        mean_path = '../savemodel/overall_mean.model'
        p_U_path = '../savemodel/p_U.model'
        p_Q_path = '../savemodel/p_Q.model'
        U_M_path = '../savemodel/U_Maxtrix.model'
        Q_M_path = '../savemodel/Q_Maxtrix.model'
        mean_f = open(mean_path, 'w')
        p_U_f = open(p_U_path, 'w')
        p_Q_f = open(p_Q_path, 'w')
        U_M_f = open(U_M_path, 'w')
        Q_M_f = open(Q_M_path, 'w')

        print self.p_U[1]
        print self.overall_mean
        mean_f.write(str(self.overall_mean))
        mean_f.close()

        for x in range(self._u_num):
            p_U_f.write(str(self.p_U[x][0]) + '\n')
            U_M_str = ""
            for k in range(self._f_num):
                if not k == self._f_num - 1:
                    U_M_str = U_M_str + str(U_Maxtrix[x][k]) + ','
                else:
                    U_M_str = U_M_str + str(U_Maxtrix[x][k]) + '\n'
            U_M_f.write(U_M_str)


        for x in range(self._q_num):
            p_Q_f.write(str(self.p_Q[x][0]) + '\n')
            Q_M_str = ""
            for k in range(self._f_num):
                if not k == self._f_num - 1:
                    Q_M_str = Q_M_str + str(Q_Maxtrix[x][k]) + ','
                else:
                    Q_M_str = Q_M_str + str(Q_Maxtrix[x][k]) + '\n'
                Q_M_f.write(Q_M_str)
        p_U_f.close()
        p_Q_f.close()
        U_M_f.close()
        Q_M_f.close()



    def predict(self, user, question):
        score = self.overall_mean + self.p_U[user] + self.p_Q[question] + np.dot(self.U_Maxtrix[user], self.Q_Maxtrix[question].T)
        score = max(score, 0.)
        score = min(score, 1.)
        return float(score)


# MF = MX_F()
# MF.load_data('../data/train_hash_shuff.txt', '\t')
# start_time = time.clock()
# MF.factorize(k = 30, iter=200, alpha = 0.01, beta = 0.05)
# end_time = time.clock()
#
# print MF.predict(2491,10)
#
# print end_time - start_time
