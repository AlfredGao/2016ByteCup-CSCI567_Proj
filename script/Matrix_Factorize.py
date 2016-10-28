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
            self.U_Maxtrix = 0.025*np.random.random((self._u_num, self._f_num))/np.sqrt(self._f_num)
            print self.U_Maxtrix.shape


    def _init_Q_Matrix(self):
        """
        Init the question matrix for matrix facorize
        """

        if self._q_num == -1 or self._f_num == -1:
            raise ValueError('Error: Miss the num of question and num of feature!')
        else:
            self.Q_Maxtrix = 0.025*np.random.random((self._q_num, self._f_num,))/np.sqrt(self._f_num)
            print self.Q_Maxtrix.shape


    def gradient_descent(self, alpha, beta):
        """
        Apply gradient descent to matrix factorize
        """
        U_M = self.U_Maxtrix
        Q_M = self.Q_Maxtrix
        penalty_U = self.p_U
        penalty_Q = self.p_Q

        #TODO What is mask???
        # self.mask = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( (np.ones(len(self.pd_train)) ,( np.array(self.pd_train['uuid']),np.array(self.pd_train['qid']) ) ) ) ).toarray()
        mask = self.mask
        mse = (self.data - np.dot(U_M, Q_M.T) - self.overall_mean - penalty_U.repeat(self._q_num, axis = 1) - penalty_Q.repeat(self._u_num, axis = 1).T)
        # print mse.shape
        # print mask.shape
        self.U_Maxtrix = U_M + alpha*(np.dot(mask*mse, Q_M) - beta*U_M)
        self.Q_Maxtrix = Q_M + alpha*(np.dot((mask*mse).T, U_M) - beta*Q_M)
        self.p_U = penalty_U + alpha*((mask*mse).sum(axis=1).reshape(penalty_U.shape) - beta*penalty_U)
        self.p_Q = penalty_Q + alpha*((mask*mse).sum(axis=0).reshape(penalty_Q.shape) - beta*penalty_Q)
        return ((mask*mse)**2).sum()



    def factorize(self, iter, k, alpha, beta):
        self._u_num = self.pd_train['uuid'].max() + 1
        self._q_num = self.pd_train['qid'].max() + 1
        self._f_num = k

        self._init_Q_Matrix()
        self._init_U_Matrix()
        # print 'row max is',self._data.row_max + 1
        # print 'row max is', self.pd_train['uuid'].max() + 1
        # print 'col max is',self._data.col_max + 1
        # print 'col max is', self.pd_train['qid'].max() + 1
        #TODO what is fuck of following??
        # mask_test =  scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( (np.ones(len(self.pd_train)) ,( np.array(self.pd_train['uuid']),np.array(self.pd_train['qid']) ) ) ) ).toarray()
        self.data = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( (np.array(self.pd_train['is_answear']),(np.array(self.pd_train['uuid']),np.array(self.pd_train['qid'])) ) )).toarray()
        # print self.data
        # self.data = self._data.get_in_numpy_format()
        # print self.data
        # test_data = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( np.array(self.pd_train['is_answear']),(np.array(self.pd_train['uuid']),np.array(self.pd_train['qid'])) ) )
        self.mask = scipy.sparse.csr_matrix( scipy.sparse.coo_matrix( (np.ones(len(self.pd_train)) ,( np.array(self.pd_train['uuid']),np.array(self.pd_train['qid']) ) ) ) ).toarray()
        # self.mask = self._data.get_mask()
        # mask_test = self._data.get_mask()
        # print 'mask test', mask_test.shape
        # print 'not test',self.mask.sum()
        # print 'test',mask_test.sum()
        self.overall_mean = self.data.sum()/self.mask.sum()
        self.p_U = np.zeros((self._u_num, 1))
        self.p_Q = np.zeros((self._q_num, 1))

        # cost = self.gradient_descent(alpha, beta)
        for i in range(iter):
            cost = self.gradient_descent(alpha, beta)
            print 'Iteration:' + str(i + 1) + ': cost is ' + str(cost)


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
