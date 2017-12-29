import pandas as pd
import numpy as np
import pyC45
from ID3Classifier import ID3Classifier
from C45Classifier import c45classifier
from NBClassT import NaiveBaiyes
from KNNClassifier import KNNClassifier
from CARTClassifier import CARTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from datetime import datetime
import warnings
import sklearn.exceptions
import itertools
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class Classification:

    #Function to get feature columns based on the datasets and algorithms specified
    def get_feature_cols(dataset, nc):
        if dataset == 'nc':
            nc_drop = ['state', "stop_date", "stop_time", "id", "violation_raw", "driver_age_raw", "driver_race_raw", "search_type_raw",
                       "location_raw", 'fine_grained_location', "county_name",'search_basis', 'officer_id', 'district', 'drugs_related_stop','county_fips']
            #selected = ['violation', 'is_arrested', 'ethnicity', 'search_type']
            bool_cols = ['search_conducted', 'contraband_found', 'is_arrested']
            float_cols = ['driver_age']

        elif dataset == 'wa':
            nc_drop = ['state', 'contact_type', 'enforcements', 'id', 'stop_date',
                       'stop_time','location_raw', 'county_name', 'fine_grained_location', 'driver_age_raw', 'driver_race_raw', 'officer_race',
                       'violation_raw', 'violation', 'search_type_raw', 'is_arrested', 'violations', 'road_number', 'police_department', 'is_arrested', 'county_fips']
            #selected = ['highway_type', 'officer_gender', 'search_type']
            float_cols = ['driver_age', 'officer_id', 'milepost', 'lat', 'lon']
            bool_cols = ['search_conducted', 'contraband_found', 'drugs_related_stop']

        else:
            nc_drop = ['state', 'id', 'stop_date', 'stop_time', 'location_raw','fine_grained_location', "county_name", 'officer_race',
                       "violation_raw", "driver_age_raw", "driver_race_raw", 'violation', 'search_type', 'search_type_raw', 'county_fips',
                      'road_number']
            # selected = ['stop_purpose', 'highway_type', 'officer_age', "county_fips",'officer_id', 'lat', 'lon']
            float_cols = ['officer_age', 'driver_age','officer_id', 'lat', 'lon']
            bool_cols = ['search_conducted', 'contraband_found', 'is_arrested']

        features = [col for col in nc.columns if col not in nc_drop]
        tocat_notbool = [col for col in features if col not in bool_cols]
        tocat = [col for col in tocat_notbool if col not in float_cols]
        one_hot_features = [col for col in tocat if col not in ['stop_outcome']]

        return nc_drop, bool_cols, tocat, one_hot_features

    def get_feature_cols_cat(dataset, nc):
        if dataset == 'nc':
            nc_drop = ['search_type', 'violation', 'search_basis', 'district', 'drugs_related_stop', 'officer_id',
                       'state', 'county_fips', 'county_name', "stop_date", "stop_time", "fine_grained_location", "id",
                       "violation_raw", "driver_age_raw", "driver_race_raw", "search_type_raw", "location_raw"]
            # selected = ['violation', 'is_arrested', 'ethnicity', 'search_type']
            bool_cols = ['search_conducted', 'contraband_found', 'is_arrested']
            float_cols = ['driver_age']

        elif dataset == 'wa':
            nc_drop = ['state', 'contact_type', 'enforcements', 'id', 'stop_date',
                       'stop_time', 'location_raw', 'county_name', 'fine_grained_location', 'driver_age_raw',
                       'driver_race_raw', 'officer_race', 'violation_raw', 'violation', 'search_type_raw', 'is_arrested',
                       'violations', 'road_number','police_department', 'is_arrested', 'county_fips', 'drugs_related_stop',
                       'officer_id', 'search_type','milepost', 'highway_type', 'lat', 'lon']
            # selected = ['highway_type', 'officer_gender', 'search_type']
            float_cols = ['driver_age', 'milepost', 'lat', 'lon']
            bool_cols = ['search_conducted', 'contraband_found']

        else:
            nc_drop = ['state', 'id', 'stop_date', 'stop_time', 'location_raw', 'fine_grained_location', "county_name",
                       'officer_race', "violation_raw", "driver_age_raw", "driver_race_raw", 'violation', 'search_type',
                       'search_type_raw', 'county_fips', 'officer_id',
                       'road_number']
            # selected = ['stop_purpose', 'highway_type', 'officer_age', "county_fips",'officer_id', 'lat', 'lon']
            float_cols = ['officer_age', 'driver_age', 'lat', 'lon']
            bool_cols = ['search_conducted', 'contraband_found', 'is_arrested']

        features = [col for col in nc.columns if col not in nc_drop]
        tocat_notbool = [col for col in features if col not in bool_cols]
        tocat = [col for col in tocat_notbool if col not in float_cols]
        one_hot_features = [col for col in tocat if col not in ['stop_outcome']]
        return nc_drop, bool_cols, tocat, one_hot_features

    #This function does the initial preprocessing
    def prepare_data(nc, nc_drop, bool_cols, one_hot_features, tocat, type = 'nondt', numrecords = 10000):
        nc = nc.drop(nc_drop, axis=1)
        nc = nc.head(numrecords)
        nc = nc.fillna(method='ffill')
        print(nc.shape)
        nc['driver_race'] = nc['driver_race'].replace({'Other': 'Other_race'})

        #To avoid duplicate indices
        if 'officer_gender' in nc.columns:
            nc['officer_gender'] = nc['officer_gender'].replace({'M': 'wa_M'})
            nc['officer_gender'] = nc['officer_gender'].replace({'F': 'wa_F'})

        print('changing bool cols to int')
        if type == 'nondt':
            for i in bool_cols:
                nc[i] = nc[i].astype(int)

        print("starting category changes..")

        for i in tocat:
            nc[i] = nc[i].astype('category')

        print("changed to category..")

        #For Naive Bayes and KNN
        if type == 'nondt':
            print("starting 1-hot ..")
            for col in one_hot_features:
                col_dummies = pd.get_dummies(nc[col])
                nc = pd.concat([nc, col_dummies], axis=1)
                del nc[col]

            print("1-hot completed..")

        nc['outcode'] = nc['stop_outcome'].cat.codes

        cols = [col for col in nc.columns if col not in ['stop_outcome']]
        nc_df = nc[cols]

        print("splitting dataset..")
        y = nc_df["outcode"]
        X = nc_df.drop(["outcode"], axis=1)

        return X, y, nc_df

    #shuffle and split the dataset into training and test sets
    def get_train_test_sets(X, y, nc_df):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        attribute_names = list(nc_df.columns)
        attribute_names.remove('outcode')

        print("split data..")
        return X_train, X_test, y_train, y_test, attribute_names

    #Naive Bayes function
    def nb(X_train, y_train, X_test, y_test):
        print("started Naive Baiyes..")
        model = NaiveBaiyes()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print ('NB Accuracy is ' + str(sum(y_test == y_pred['predicted']) / float(len(y_test))))
        return y_pred['predicted']

    #ID3 function
    def id3(X_train, y_train, X_test, y_test, attribute_names, default_class):
        print("starting id3..")
        # for id3
        model = ID3Classifier()
        test_data = pd.concat([X_test, y_test], axis=1)
        train_set = pd.concat([X_train, y_train], axis=1)
        print("building tree..")
        train_tree = model.id3(train_set, 'outcode', attribute_names, default_class)
        res = test_data.apply(model.classify, axis=1, args=(train_tree, 1))
        print ('ID3 accuracy is ' + str( sum(test_data['outcode']==res ) /(1.0*len(test_data))))
        return res

    #KNN function
    def knn(X_train, y_train, X_test, y_test):
        print("starting knn..")
        model = KNNClassifier(X_train, X_test, y_train, y_test)
        res = model.knn()
        print ('KNN Accuracy is ' + str(sum(y_test == res) / float(len(y_test))))
        return res

    #CART function
    def cart(X_train, y_train, X_test, y_test):
        print("starting cart..")
        test_data = pd.concat([X_test, y_test], axis=1)
        train_set = pd.concat([X_test, y_test], axis=1)
        test_dataset = test_data.values.tolist()
        train_dataset = train_set.values.tolist()
        model = CARTClassifier(train_dataset,test_dataset)
        predicted = model.decision_tree(train_dataset, test_dataset, 5, 10)
        actual = [row[-1] for row in test_dataset]
        accuracy = model.accuracy_metric(actual, predicted)
        print ('CART Accuracy is ' + str(accuracy))
        return predicted

    def c45(X_train, y_train, X_test, y_test):
        pyC45.train(X_train, y_train, "DecisionTree.xml")

        # test the C45 decision tree
        answer = []
        testing_obs = []
        for index, row in y_test.iteritems():
            # testing_obs.append(row[:-1].tolist())
            answer.append(str(row))
        prediction = pyC45.predict("DecisionTree.xml", X_test)
        return answer, prediction

    def drop_non_unique(nc):
        drop_columns = []
        for column in nc.columns:
            non_null = nc[column].dropna()
            unique_non_null = non_null.unique()
            if len(unique_non_null) == 1:
                drop_columns.append(column)

        nc.drop(drop_columns, axis=1, inplace = True)
        return nc

    #Method to get performance metrics for the predictions
    def accuracy_metrics(testY, result_df):
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

    #This plots the confusion matrices
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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

    #The main function
    if __name__ == "__main__":
        print("reading datasets..")
        #North Carolina dataset
        nc = pd.read_csv("NC.csv")

        #South Carolina dataset
        # sc = pd.read_csv("SC.csv")
        #
        # #Washington dataset
        # wa = pd.read_csv("WA.csv")
        # wa.dropna(subset=['driver_gender'], inplace=True)
        # print("read datasets")

        #List of datasets on which algo has to be run
        # datasets = ['nc', 'sc', 'wa'] #update this
        datasets = ['nc']

        #List of algos to be run
        # algos = ['nb', 'id3', 'knn', 'cart', 'c45'] #update this
        algos = ['c45']

        #DataFrames for the datasets
        # dfs = [nc, sc, wa] #update this
        dfs = [nc]

        numrecords = 10000

        for i, data in enumerate(datasets):

            for algo in algos:
                start_time = datetime.now()
                print('algo ------------------------------')
                if algo == 'nb':
                    nc_drop, bool_cols, tocat, one_hot_features = get_feature_cols(data, dfs[i])
                    X, y, df = prepare_data(dfs[i], nc_drop, bool_cols, one_hot_features, tocat, numrecords)
                    X_train, X_test, y_train, y_test, attribute_names = get_train_test_sets(X, y, df)
                    ypred = nb(X_train, y_train, X_test, y_test)

                elif algo == 'id3':
                    nc_drop, bool_cols, tocat, one_hot_features = get_feature_cols_cat(data, dfs[i])
                    X, y, df = prepare_data(dfs[i], nc_drop, bool_cols, one_hot_features, tocat, 'dt', numrecords)
                    X_train, X_test, y_train, y_test, attribute_names = get_train_test_sets(X, y, df)
                    default_class = y_train.value_counts().index[0]
                    ypred = id3(X_train, y_train, X_test, y_test, attribute_names, default_class)
                    ypred = ypred.fillna(default_class)

                elif algo == 'knn':
                    nc_drop, bool_cols, tocat, one_hot_features = get_feature_cols(data, dfs[i])
                    X, y, df = prepare_data(dfs[i], nc_drop, bool_cols, one_hot_features, tocat, numrecords)
                    X_train, X_test, y_train, y_test, attribute_names = get_train_test_sets(X, y, df)
                    ypred = knn(X_train, y_train, X_test, y_test)


                elif algo == 'cart':
                    nc_drop, bool_cols, tocat, one_hot_features = get_feature_cols_cat(data, dfs[i])
                    X, y, df = prepare_data(dfs[i], nc_drop, bool_cols, one_hot_features, tocat, 'dt', numrecords)
                    X_train, X_test, y_train, y_test, attribute_names = get_train_test_sets(X, y, df)
                    default_class = y_train.value_counts().index[0]
                    ypred = cart(X_train, y_train, X_test, y_test)
                    # ypred = ypred.fillna(default_class)

                elif algo == 'c45':
                    nc_drop, _, tocat, _ = get_feature_cols_cat(data, dfs[i])
                    c45 = c45classifier(dfs[i])
                    c45.classify(nc_drop, tocat, 4000)

                if algo != 'c45':
                    dt = datetime.now() - start_time
                    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
                    print ('Time taken by this algo in millisec:' + str(ms))
                    acc, class_names, cnf_matrix = accuracy_metrics(y_test, ypred)
                print('algo end----------------------------')









