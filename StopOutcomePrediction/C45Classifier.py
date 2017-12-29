import pandas as pd
import pyC45
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import itertools
import numpy as np

class c45classifier:

    def __init__(self, df):
        self.df = df

    def accuracy_metrics(self, testY, result_df):
        macro_prec_score = precision_score(testY, result_df, average='macro')
        micro_prec_score = precision_score(testY, result_df, average='micro')
        wt_prec_score = precision_score(testY, result_df, average='weighted')
        print('Macro Precision score: {0:0.2f}'.format(macro_prec_score))
        print('Micro Precision score: {0:0.2f}'.format(micro_prec_score))
        print('Weighted Precision Score: {0:0.2f}'.format(wt_prec_score))
        recall = recall_score(testY, result_df, average='macro')
        print('Recall score: {0:0.2f}'.format(recall))
        f1_sc = f1_score(testY, result_df, average='macro')
        print('F1 score: {0:0.2f}'.format(f1_sc))
        from sklearn.metrics import matthews_corrcoef
        math_coef = matthews_corrcoef(testY, result_df)
        print('Mathews coeff {0:0.2f}'.format(math_coef))
        cnf_matrix = confusion_matrix(testY, result_df)
        class_names = testY.unique()

        # For each class
        acc = accuracy_score(testY, result_df)
        print('Accuracy score: {0:0.2f}'.format(acc))
        return acc, class_names, cnf_matrix


    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def preprocess(self, nc_drop, tocat, numrecords = 10000):
        nc = self.df
        nc = nc.head(numrecords)
        nc = nc.drop(nc_drop, axis=1)
        nc = nc.fillna(method='ffill')


        if 'officer_gender' in nc.columns:
            nc['officer_gender'] = nc['officer_gender'].replace({'M': 'wa_M'})
            nc['officer_gender'] = nc['officer_gender'].replace({'F': 'wa_F'})

        nc['driver_race'] = nc['driver_race'].replace({'Other': 'Other_race'})
        for i in tocat:
            nc[i] = nc[i].astype('category')

        nc['outcode'] = nc['stop_outcome'].cat.codes

        cols = [col for col in nc.columns if col not in ['outcode', 'stop_outcome']]
        train = nc[cols]
        test = nc["outcode"]
        trainX = train.iloc[:int(len(nc) * 0.8), :]
        testX = train.iloc[int(len(nc) * 0.8):, :]

        trainY = test[:int(len(nc) * 0.8)]
        testY = test[int(len(nc) * 0.8):]

        return trainX, trainY, testX, testY

    def trainandpredict(self, trainX, trainY, testX, testY):
        startTime = datetime.now()
        pyC45.train(trainX, trainY, "DecisionTree.xml")
        start_time = datetime.now()

        # test the C45 decision tree
        answer = []
        testing_obs = []
        for index, row in testY.iteritems():
            answer.append(str(row))
        startTime = datetime.now()
        prediction = pyC45.predict("DecisionTree.xml", testX)
        predictionTime = datetime.now()
        dt = datetime.now() - start_time
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        print ('Time taken by this algo in millisec:' + str(ms))

        return answer, prediction

    def printperformance(self, answer, prediction):
        acc, class_names, cnf_matrix = self.accuracy_metrics(pd.Series(answer), pd.Series(prediction))

    def classify(self, nc_drop, tocat, numrecords):
        trainX, trainY, testX, testY = self.preprocess(nc_drop, tocat, numrecords)
        answer, prediction = self.trainandpredict(trainX, trainY, testX, testY)
        self.printperformance(answer, prediction)

