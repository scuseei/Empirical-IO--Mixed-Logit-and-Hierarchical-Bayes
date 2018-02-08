# -*- coding: utf-8 -*-
"""
Task
-------
Mixed Logit and Hierarchical Bayes

Version      |Author       |Affiliation                |Email
--------------------------------------------------------------------------
Feb 10, 2018 |Chenshuo Sun |Stern Business School, NYU |csun@stern.nyu.edu

Goal(s)
-------
Simulate data sets
"""

import numpy as np
import pandas as pd


class DataSimulator(object):
    """Class for data simulating

    Parameters
    ----------
    param1: int
        Number of consumers I
    param2: int
        Number of consumers J
    param3: int
        Number of consumers T
    param4: list
        Distribution of prices f(pjt) to simulate prices, e.g., ['Normal', 1, 0.3]
    param5: float64
        True parameter values for brand intercept, e.g., [0, 0] for two brands
    param6: float64
        Distribution of price sensitivity \beta, e.g., ['Normal', -1.5, 0.4]
    param7: int
        Simulate M data sets, e.g., M = 10

    Returns
    -------
    DataFrame
        Brand chosen
    DataFrame
        Prices, for each consumer i, brand j, and time t
    """

    def __init__(self, I, J, T, pdf_price, intercept, pdf_coefficient, M):
        """Class initialization"""
        self.I = I
        self.J = J
        self.T = T
        self.pdf_price = pdf_price
        self.intercept = intercept
        self.pdf_coefficient = pdf_coefficient
        self.M = M

    def price(self):
        """Function for simulating the price for each (j, t)

        Parameters
        ----------
        param1: int
            Number of consumers J
        param2: int
            Number of consumers T
        param3: list
            Distribution of prices f(pjt) to simulate prices, e.g., ['Normal', 1, 0.3]

        Returns
        -------
        DataFrame
            Price for each brand j at time t
        """
        if self.pdf_price[0] == 'Normal':
            p_mean = self.pdf_price[1]
            p_std = self.pdf_price[2]
        price = pd.DataFrame(
            index=range(
                self.J * self.T),
            columns=[
                'J',
                'T',
                'Pjt']).fillna(0.0)
        for j in range(self.J):
            for t in range(self.T):
                price.iloc[t + j * self.T]['J'] = j
                price.iloc[t + j * self.T]['T'] = t
                price.iloc[t + j *
                           self.T]['Pjt'] = np.random.normal(p_mean, p_std, 1)
        price['J'] += 1
        price['T'] += 1
        price['J'] = price['J'].astype('int')
        price['T'] = price['T'].astype('int')
        # return
        return price

    def choice(self):
        """Function for simulating the utility for each (i, t)

        Parameters
        ----------
        param1: int
            Number of consumers I
        param2: int
            Number of consumers J
        param3: int
            Number of consumers T

        Returns
        -------
        DataFrame
            Data
        """
        price = self.price()
        b_mean = self.pdf_coefficient[1]
        b_std = self.pdf_coefficient[2]

        '''Generate choice probability'''
        Y = pd.concat([price[['T', 'J', 'Pjt']]]
                      * self.I, ignore_index=False)
        Y['I'] = pd.Series(0, index=Y.index, dtype='int')
        Y['I'] = np.repeat(
            np.linspace(
                1,
                self.I,
                self.I,
                dtype=int),
            self.J *
            self.T)
        Y['Bi'] = pd.Series(0, index=Y.index, dtype='int')
        Y['Bi'] = np.repeat(
            np.random.normal(
                b_mean,
                b_std,
                self.I),
            self.J *
            self.T)
        Y['Pr(j)'] = pd.Series(0, index=Y.index, dtype='int')
        Y['Yitj'] = pd.Series(0, index=Y.index, dtype='int')
        Y = Y[['I', 'T', 'J', 'Pjt', 'Bi', 'Pr(j)', 'Yitj']]
        Y = Y.sort_values(['I', 'T'], ascending=[True, True]
                          ).reset_index(drop=True)
        Z = Y.set_index(['I', 'T', 'J'])
        W = Z.groupby(['I', 'T']).size().index.values
        # U = []
        for (i, t) in W:
            temp_sum = 0
            for j in range(1, self.J + 1):
                temp_sum += np.exp(self.intercept[j - 1] +
                                   Z.loc[(i, t)].iloc[j - 1]['Bi'] *
                                   Z.loc[(i, t)].iloc[j - 1]['Pjt'])
            for j in range(1, self.J + 1):
                Z.loc[(i, t, j), 'Pr(j)'] = np.exp(self.intercept[j - 1] +
                                                   Z.loc[(i, t)].iloc[j - 1]['Bi'] *
                                                   Z.loc[(i, t)].iloc[j - 1]['Pjt']) / (1 + temp_sum)
            # print(i)
            temp_u = np.random.uniform(0, 1, 1)
            # U.append(temp_u)
            temp_v = Z.loc[(i, t, 1), 'Pr(j)']
            if temp_u <= temp_v:
                Z.loc[(i, t, 1), 'Yitj'] = 1
            else:
                for j in range(2, self.J + 1):
                    temp_v += Z.loc[(i, t, j), 'Pr(j)']
                    if temp_u <= temp_v:
                        Z.loc[(i, t, j), 'Yitj'] = 1
                        break
        Z = Z.reset_index()
        Z['Yitj'] = Z['Yitj'] * Z['J']
        # write up for output
        for j in range(self.J + 1):
            choice_name = 'Yit' + str(j)
            Z[choice_name] = pd.Series(0, index=Z.index, dtype='int')
        P = Z.set_index(['I', 'T'])
        for (i, t) in W:
            temp_choice = np.sum(P.loc[(i, t)]['Yitj'])
            temp_choice_name = 'Yit' + str(temp_choice)
            P.loc[(i, t), temp_choice_name] = 1
        P = P.reset_index()
        P = P.drop('Yitj', 1)
        # return    
        return P

    def make_data(self):
        """Function for simulating the utility for each (i, t)

        Parameters
        ----------
        param1: Simulate M data sets, e.g., M = 10

        Returns
        -------
        DataFrame
            Data stacked
        """
        M = self.M
        for m in range(M):
            file_name = 'Data_' + str(m + 1) + '.csv'
            data = self.choice()
            data.to_csv(file_name)


def main():
    """Main function for calculating the lift and leverage value (results)
    """
    '''set parameters'''
    I = 500
    J = 2
    T = 20
    pdf_price = ['Normal', 1, 0.3]
    intercept = np.zeros(J)
    pdf_coefficient = ['Normal', -1.5, 0.4]
    M = 10

    '''get simulation data'''
    ds = DataSimulator(
        I, J, T, pdf_price, intercept, pdf_coefficient, M)
    ds.make_data()


if __name__ == "__main__":
    main()
