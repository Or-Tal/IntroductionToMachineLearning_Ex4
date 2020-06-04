"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def _update_distribution(self, w, y, D, predictions):
        """
        Parameters
        ----------
        w : weight of current base classifier
        y : labels, shape=(num_samples)
        D : sample weights, shape=(num_samples)
        predictions : the model's predictions, shape (num_samples)

        return
        ----------
        weighted zero-one loss over samples
        """
        # define vectors
        denom = D * np.exp(((-w) * y * predictions))
        devisor = np.sum(denom)

        # return D
        return denom / devisor

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """

        # init distribution
        D = np.ones((X.shape[0]))/X.shape[0]  # D = [1/m,...,1/m]

        # iteratively update weights
        for i in np.arange(self.T):
            # invoke base learner
            self.h[i] = self.WL(D, X, y)

            # predict
            predictions = self.h[i].predict(X)

            # compute err
            err = np.sum(D[predictions != y])

            # update weights[i]
            self.w[i] = np.inf if err == 0 else 0.5 * np.log((1 / err) - 1)

            # update D
            if i < self.T - 1:
                D = self._update_distribution(self.w[i], y, D, predictions)
        # return last D
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """

        # get predictions
        predictions = np.zeros(X.shape[0])
        for i, c in enumerate(self.h[:max_t]):
            pred = c.predict(X)
            predictions += self.w[i] * pred
        return np.sign(predictions)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the incorrect predictions when predict only with max_t weak learners (float)
        """
        predictions = self.predict(X, max_t)

        # gen incorrect predictions count
        incorrect_num = np.where(predictions != y)[0]

        # return error
        return incorrect_num.shape[0] / y.shape[0]