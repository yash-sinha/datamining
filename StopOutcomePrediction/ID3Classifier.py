import pandas as pd
import numpy as np

class ID3Classifier():

    def entropy(self, probs):
        '''
        Takes a list of probabilities and calculates their entropy
        '''
        import math
        return sum([-prob * math.log(prob, 5) for prob in probs])

    def entropy_of_list(self, a_list):
        '''
        Takes a list of items with discrete values (e.g., poisonous, edible)
        and returns the entropy for those items.
        '''
        from collections import Counter

        # Tally Up:
        cnt = Counter(x for x in a_list)

        # Convert to Proportion
        num_instances = len(a_list) * 1.0
        probs = [x / num_instances for x in cnt.values()]

        # Calculate Entropy:
        return self.entropy(probs)

    # In[26]:


    def information_gain(self, df, split_attribute_name, target_attribute_name, trace=0):
        '''
        Takes a DataFrame of attributes, and quantifies the entropy of a target
        attribute after performing a split along the values of another attribute.
        '''

        # Split Data by Possible Vals of Attribute:
        df_split = df.groupby(split_attribute_name)

        # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split
        nobs = len(df.index) * 1.0
        df_agg_ent = df_split.agg({target_attribute_name: [self.entropy_of_list, lambda x: len(x) / nobs]})[
            target_attribute_name]
        df_agg_ent.columns = ['Entropy', 'PropObservations']
        if trace:  # helps understand what fxn is doing:
            print (df_agg_ent)

        # Calculate Information Gain:
        new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
        old_entropy = self.entropy_of_list(df[target_attribute_name])
        return old_entropy - new_entropy


    def id3(self, df, target_attribute_name, attribute_names, default_class=None):
        from collections import Counter
        cnt = Counter(x for x in df[target_attribute_name])
        cnt = dict(cnt)

        if len(cnt) == 1:
            return next(iter(cnt.keys()))

        elif df.empty or (not attribute_names):
            return default_class

        else:
            default_class = max(cnt, key=cnt.get)

            # Choose Best Attribute to split on:
            gainz = [self.information_gain(df, attr, target_attribute_name) for attr in attribute_names]
            index_of_max = gainz.index(max(gainz))
            best_attr = attribute_names[index_of_max]

            # Create an empty tree, to be populated in a moment
            tree = {best_attr: {}}
            remaining_attribute_names = [i for i in attribute_names if i != best_attr]

            # Split dataset
            # On each split, recursively call this algorithm.
            # populate the empty tree with subtrees, which
            # are the result of the recursive call
            for attr_val, data_subset in df.groupby(best_attr):
                subtree = self.id3(data_subset,
                              target_attribute_name,
                              remaining_attribute_names,
                              default_class)
                tree[best_attr][attr_val] = subtree
            return tree

    def classify(self, instance, tree, default=None):
        attribute = next(iter(tree.keys()))
        if instance[attribute] in tree[attribute].keys():
            result = tree[attribute][instance[attribute]]
            if isinstance(result, dict): # this is a tree, delve deeper
                return self.classify(instance, result)
            else:
                return result # this is a label
        else:
            return default

