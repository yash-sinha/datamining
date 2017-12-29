import numpy as np
import math
import pandas as pd
class NaiveBaiyes:
    """The Gaussian Naive Bayes classifier. """

    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = y.unique()
        # self.parameters = []
        self.prior = []

        self.means = []
        self.vars = []

        # Calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            # Only select the rows where the label equals the given class
            X_where_c = X[y == c]#X[np.where(y == c)]

            n_class_instances = X_where_c.shape[0]
            n_total_instances = self.X.shape[0]
            prior =  n_class_instances / float(n_total_instances)
            # self.prior.append([])
            # self.means.append([])
            # self.vars.append([])

            self.prior.append(prior)
            mean = X_where_c.mean()
            var = X_where_c.var()
            self.means.append(X_where_c.mean())
            self.vars.append(X_where_c.var())
            # Add the mean and variance for each feature (column)
            # for j in range(X.shape[1]):
            #     col = X_where_c.iloc[:, j]
            #     self.means[i].append[col.mean()]
            #     self.vars[i].append[col.var()]
                # parameters = {"mean": col.mean(), "var": col.var()}
                # self.parameters[i].append(parameters)

    # def shuffle_data(X, y, seed=None):
    #     """ Random shuffle of the samples in X and y """
    #     if seed:
    #         np.random.seed(seed)
    #     idx = np.arange(X.shape[0])
    #     np.random.shuffle(idx)
    #     return X[idx], y[idx]

    # def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    #     """ Split the data into train and test sets """
    #     if shuffle:
    #         X, y = X.shuffle_data(X, y, seed)
    #     # Split the training data from test data in the ratio specified in
    #     # test_size
    #     split_i = len(y) - int(len(y) // (1 / test_size))
    #     X_train, X_test = X[:split_i], X[split_i:]
    #     y_train, y_test = y[:split_i], y[split_i:]
    #
    #     return X_train, X_test, y_train, y_test

    def _calculate_likelihood(self, mean, var, df):
        """ Gaussian likelihood of the data x given mean and var """
        # eps = 1e-4  # Added in denominator to prevent division by zero
        # coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        # exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        # return coeff * exponent

        eps = 1e-4  # Added in denominator to prevent division by zero
        coeff = 1.0 / np.sqrt(2.0 * math.pi * var + eps)
        # print(var.shape)
        # print(df.shape)
        exponent = np.exp(-np.square(df.sub(df.mean()))/(2 * var + eps))
        df = coeff * exponent
        return df.product(axis = 1)

    def accuracy_score(y_true, y_pred):
        """ Compare y_true to y_pred and return the accuracy """
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    # def normalize(X, axis=-1, order=2):
    #     """ Normalize the dataset X """
    #     l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    #     l2[l2 == 0] = 1
    #     return X / np.expand_dims(l2, axis)

    def _calculate_prior(self, c):
        """ Calculate the prior of class c
        (samples where class == c / total number of samples)"""
        X_where_c = self.X[self.y == c]
        n_class_instances = X_where_c.shape[0]
        n_total_instances = self.X.shape[0]
        return n_class_instances / float(n_total_instances)

    def _classify(self, df):
        """ Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X)
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _calculate_likelihood)
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        # Go through list of classes
        res_df = pd.DataFrame()
        for i, c in enumerate(self.classes):
            posterior = self.prior[i] #self._calculate_prior(c)
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # Multiply with the class likelihoods
            likelihood = self._calculate_likelihood(self.means[i], self.vars[i], df)

            posterior *= likelihood

            # for j in enumerate(self.cl):
            #     sample_feature = sample[j]
                # Determine P(x|Y)

                # Multiply with the accumulated probability

            # Total posterior = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
            res_df = pd.concat([res_df, posterior], axis = 1)
            # posteriors.append(posterior)
        # Return the class with the largest posterior probability
        # index_of_max = np.argmax(posteriors)
        # print(res_df.head)
        res_df.columns = self.classes
        return res_df.idxmax(axis=1)

    def predict(self, X):
        """ Predict the class labels of the samples in X """
        y = pd.Series()
        y['predicted'] = self._classify(X) #X.apply(self._classify,axis=1)
        # for sample in X:
        #     y = self._classify(sample)
        #     y_pred.append(y)
        return y
