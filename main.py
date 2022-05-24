import pandas as pd
import xgboost as xgb

from preprocessing import preprocessing_1
from sklearn.datasets import *


class Lambda_Ranker:

    def __init__(self, TRAIN_DATA, TEST_DATA=None):

        self.training_data = pd.read_csv(TRAIN_DATA, nrows=10000)
        self.validation_data = pd.read_csv(TRAIN_DATA, skiprows=20000, nrows=10000, header=None,
                                           names=self.training_data.columns)
        self.test_data = pd.read_csv(TRAIN_DATA, skiprows=100000, nrows=20000, header=None,
                                     names=self.training_data.columns)

        preprocessing_1(self.training_data)
        preprocessing_1(self.validation_data)
        preprocessing_1(self.test_data)

        self.dump_files()
        self.train_LambdaMART()

    @staticmethod
    def relevance(a):
        if a[0] == a[1] == 1:
            return 5
        elif a[0] == 1 and a[1] == 0:
            return 1
        else:
            return 0

    def dump_files(self):

        dump_svmlight_file(self.training_data.iloc[:, 1:-2].values,
                           self.training_data.iloc[:, -2:].apply(self.relevance, axis=1),
                           'data/svmlight_training.txt', query_id=self.training_data.srch_id)

        dump_svmlight_file(self.validation_data.iloc[:, 1:-2].values,
                           self.validation_data.iloc[:, -2:].apply(self.relevance, axis=1),
                           'data/svmlight_validation.txt', query_id=self.validation_data.srch_id)

        dump_svmlight_file(self.test_data.iloc[:, 1:-2].values,
                           self.test_data.iloc[:, -2:].apply(self.relevance, axis=1),
                           'data/svmlight_test.txt', query_id=self.test_data.srch_id)

    def train_LambdaMART(self):
        model = xgb.XGBRanker(tree_method='gpu_hist',
                              booster='gbtree',
                              objective='rank:pairwise',
                              random_state=42,
                              learning_rate=0.1,
                              colsample_bytree=0.9,
                              eta=0.05,
                              max_depth=2,
                              n_estimators=10000,
                              subsample=0.75
                              )
        model = LambdaMART(self.training_data)
        model.fit(X_train, y_train, group=groups, verbose=True)

        k = 5
        ndcg = model.validate(self.test_data, k)
        print(ndcg)



training_path = "original_data/training_set_VU_DM.csv"
testing_path = "original_data/test_set_VU_DM.csv"

short_train = "Group118_Assn2/data/5000/training_data_5000.csv"
short_test = "Group118_Assn2/data/2500/testing_data_2500.csv"
Lambda_Ranker(short_train)
