import pandas as pd
import numpy as np
from scipy.spatial import distance


class KNNClassifier():
    def __init__(self, trainX, testX, trainY, testY):
        self.testX = testX
        self.trainX = trainX
        self.trainY = trainY
        self.testY = testY

    def knn2(self, input_data, training_set, labels, k=1):
        distance_diff = training_set - input_data
        distance_squared = distance_diff ** 2
        distance = distance_squared.sum(axis=1) ** 0.5
        distance_df = pd.concat([distance, labels], axis=1)
        colname = list(distance_df)[0]
        distance_df.sort_values(by=[colname], inplace=True)
        top_knn = distance_df[:k]
        ser = top_knn.iloc[:, 1]
        # maxfreq= top_knn[1].value_counts()
        return ser.value_counts().index.values[0]

    def knn(self):
        result_df = self.testX.apply(lambda row: self.knn2(row, self.trainX, self.trainY, k=10), axis=1)
        error_df = result_df == self.testY
        return error_df

