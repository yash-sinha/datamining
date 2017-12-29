from random import seed
from random import randrange


class CARTClassifier:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    # Split a dataset into k folds
    def cross_validation_split(self,dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split


    # Calculate accuracy percentage
    def accuracy_metric(self,actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self,dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            print("Now printing train set")
            print(train_set)
            print("Now printing test set")
            print(test_set)
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(self,actual, predicted)
            scores.append(accuracy)
        return scores


    # Calculate the Gini index for a split dataset
    def gini_index(self,groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini


    def divideset(self, rows, column, value):
        left, right = list(), list()
        # Make a function that tells us if a row is in the first group
        # (true) or the second group (false)
        split_function = None
        # for numerical values
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        # for nominal values
        else:
            split_function = lambda row: row[column] == value
            # Divide the rows into two sets and return them
        set1 = [row for row in rows if split_function(row)]  # if split_function(row)
        print("Printing set1")
        print(set1)
        left.append(set1)
        set2 = [row for row in rows if not split_function(row)]
        right.append(set2)
        return left, right


    # Select the best split point for a dataset
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                #groups = self.divideset(dataset, index, row[index])
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}



    # Split a dataset based on an attribute and an attribute value
    def test_split(self,index, value, dataset):
        left, right = list(), list()
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[index] < value
        # for nominal values
        else:
            split_function = lambda row: row[index] == value
        for row in dataset:
            if split_function(row):
                left.append(row)
            else:
                right.append(row)
        return left, right


    # Create a terminal node value
    def to_terminal(self,group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)


    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)


    # Build a decision tree
    def build_tree(self, train, max_depth, min_size):
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root


    # Make a prediction with a decision tree
    def predict(self,node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']


    # Classification and Regression Tree Algorithm
    def decision_tree(self, train, test, max_depth, min_size):
        tree = self.build_tree(train, max_depth, min_size)
        print(tree)
        predictions = list()
        for row in test:
            prediction = self.predict(tree, row)
            predictions.append(prediction)
        return (predictions)
