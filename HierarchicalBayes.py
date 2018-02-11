# -*- coding: utf-8 -*-
"""
Task
-------
Nested Logit and MLE

Version      |Author       |Affiliation                |Email
--------------------------------------------------------------------------
Feb 10, 2018 |Chenshuo Sun |Stern Business School, NYU |csun@stern.nyu.edu

Goal(s)
-------
Simulate data sets
"""

import numpy as np
import pandas as pd
from scipy.stats import invwishart


class Obj(object):
    """Class for Hierarchical Bayes Estimation
    """

    def __init__(self, m, I, J, T, R, b_mean_ini, b_std_ini):
        """Class initialization"""
        self.m = m
        self.I = I
        self.J = J
        self.T = T
        self.R = R
        self.b_mean_ini = b_mean_ini
        self.b_std_ini = b_std_ini

    def data_loader(self):
        """Function for loading the data
        """
        m = self.m
        file_name = 'Data_' + str(m) + '.csv'
        # print(file_name + ' loaded')
        data = pd.read_csv(file_name)
        return data

    def __LL(self, intercept_r, coefficient_i_r, i_):
        """Function for computing the log likelihood
        """
        # parameters initialized
        J = self.J
        data = self.data_loader()
        # compute the Pr(j)
        Z = data.set_index(['I', 'T', 'J'])
        Z['Pr(j)'] = pd.Series(0.0, index=data.index, dtype='float64')
        W = Z.groupby(['I', 'T']).size().index.values
        for (i, t) in W:
            if i == i_:
                temp_sum = 0
                for j in range(1, J + 1):
                    temp_sum += np.exp(intercept_r[j - 1] + coefficient_i_r *
                                       Z.loc[(i, t, j)]['Pjt'])
                for j in range(1, J + 1):
                    Z.loc[(i, t, j), 'Pr(j)'] = np.exp(intercept_r[j - 1] + coefficient_i_r *
                                                       Z.loc[(i, t, j)]['Pjt']) / (1 + temp_sum)
        # compute the log-likelihood, step 1
        LH_it = pd.DataFrame(index=W, columns=['lh_it'])
        for (i, t) in W:
            if i == i_:
                temp_choice = int(
                    Z.loc[(i, t, 1),
                          Z.columns.str.startswith('Yit', na=False)].nonzero()[0])
                if temp_choice == 0:
                    temp_Pr0 = 1 - Z.loc[(i, t), 'Pr(j)'].sum()
                    lh_it = temp_Pr0
                else:
                    temp_Prj = Z.loc[(i, t, temp_choice), 'Pr(j)']
                    lh_it = temp_Prj
                LH_it.loc[(i, t), 'lh_it'] = lh_it
        # compute the log-likelihood, step 2
        LH_it = LH_it.dropna()
        lh_i_ = 1
        for r in range(len(LH_it)):
            lh_i_ *= LH_it.iloc[r]['lh_it']
        # return
        # print('...')
        return lh_i_

    def __draw_b(
            self,
            coefficient_r_1,
            coefficient_mean_r_1,
            coefficient_std_r_1,
            LL_r_1):
        """Function for drawing the b_i at the r-round
        """
        I = self.I
        intercept_r = np.array([0.0, 0.0])
        coefficient_r = pd.DataFrame(index=range(
            1, I + 1), columns=['coefficient_r'])
        LL_r = pd.DataFrame(index=range(1, I + 1), columns=['LL_r'])
        for i_ in range(1, I + 1):
            coefficient_i_r = np.random.normal(
                coefficient_mean_r_1, coefficient_std_r_1, 1)
            LL_i_r = self.__LL(intercept_r, coefficient_i_r, i_)
            temp_u = np.random.uniform(0, 1, 1)
            temp_lambda = min([1, float(LL_i_r / LL_r_1.loc[i_])])
            if temp_u < temp_lambda:
                coefficient_r.loc[i_] = coefficient_i_r
                LL_r.loc[i_] = LL_i_r
            else:
                coefficient_r.loc[i_] = coefficient_r_1.loc[i_]
                LL_r.loc[i_] = LL_r_1.loc[i_]
        return coefficient_r, LL_r

    def __draw_b_mean(self, coefficient_r, coefficient_std_r_1):
        """Function for drawing the mean at the r-round
        """
        coefficient_mean_r = np.mean(coefficient_r)
        coefficient_mean_draw = np.random.normal(
            coefficient_mean_r, coefficient_std_r_1, 1)
        return coefficient_mean_draw

    def __draw_b_std(self, coefficient_r, coefficient_mean_r):
        """Function for drawing the std at the r-round
        """
        I = self.I
        df = 1 + I
        scale = (I + np.var(coefficient_r)) / df
        coefficient_std_draw = invwishart.rvs(df, scale)
        return coefficient_std_draw

    def b_iteration(self):
        """Function for the main iteration
        """
        I = self.I
        R = self.R
        coefficient = pd.DataFrame(index=range(R), columns=range(1, I + 1))
        coefficient_mean = pd.DataFrame(index=range(R), columns=['b_mean'])
        coefficient_std = pd.DataFrame(index=range(R), columns=['b_std'])
        LL_ = pd.DataFrame(index=range(R), columns=range(1, I + 1))
        # initial value
        b_mean_ini = self.b_mean_ini
        b_std_ini = self.b_std_ini
        b_ini = np.random.normal(b_mean_ini, b_std_ini, I)
        LL_ini = pd.Series(0, index=range(1, I + 1))

        for r in range(R):
            # draw b
            if r == 0:
                coefficient_r, LL_r = self.__draw_b(
                    b_ini, b_mean_ini, b_std_ini, LL_ini)
                coefficient.iloc[0, :] = coefficient_r['coefficient_r']
                LL_.iloc[r, :] = LL_r['LL_r']
                # draw b_mean
                coefficient_mean_draw = self.__draw_b_mean(
                    coefficient_r, b_std_ini)
                coefficient_mean.iloc[0] = coefficient_mean_draw
                # draw b_std
                coefficient_std_draw = self.__draw_b_std(
                    coefficient_r, coefficient_mean_draw)
                coefficient_std.iloc[0] = coefficient_std_draw
            else:
                # steps run
                coefficient_r, LL_r = self.__draw_b(
                    coefficient_r, coefficient_mean_draw,
                    coefficient_std_draw, LL_r)
                coefficient.iloc[r, :] = coefficient_r['coefficient_r']
                LL_.iloc[r, :] = LL_r['LL_r']
                # draw b_mean
                coefficient_mean_draw = self.__draw_b_mean(
                    coefficient_r, coefficient_std_draw)
                coefficient_mean.iloc[r] = coefficient_mean_draw
                # draw b_std
                coefficient_std_draw = self.__draw_b_std(
                    coefficient_r, coefficient_mean_draw)
                coefficient_std.iloc[r] = coefficient_std_draw
            print('Iteration: ' + str(r))
        return coefficient_mean, coefficient_std


def result_show(m, I, J, T, R, b_mean_ini, b_std_ini):
    """Function for showing the result
    """
    obj = Obj(m, I, J, T, R, b_mean_ini, b_std_ini)
    print('It needs some time to estimate...')
    result = obj.b_iteration()
    pos_mean = result[0].iloc[-1]
    pos_std = result[1].iloc[-1]
    print('Estimation is done on the ' + str(m) + '-th dataset')
    print('Posterior mean: ' + str(pos_mean) + '\n')
    print('Posterior sd: ' + str(pos_std) + '\n')


def main():
    """Main function
    """
    '''set parameters'''
    m = 3
    I = 500
    J = 2
    T = 20
    R = 50
    b_mean_ini = 0
    b_std_ini = 10
    # print results
    result_show(m, I, J, T, R, b_mean_ini, b_std_ini)


if __name__ == "__main__":
    main()
