import numpy as np
import pandas as pd
import lightgbm
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from preprocessing import PreProcess
from sklearn.preprocessing import StandardScaler


class Light_GBM:

    def __init__(self, filepath):
        self.df_train = pd.read_csv(filepath)
        # X_train, X_gold, y_train, y_gold = self.training_split()

        self.parameters = {'objective': 'lambdarank',
                           'metric': 'ndcg',
                           'is_unbalance': 'true',
                           'boosting': 'gbdt',
                           'num_leaves': 63,
                           'feature_fraction': 0.5,
                           'bagging_fraction': 0.5,
                           'bagging_freq': 20,
                           'learning_rate': 0.01,
                           'verbose': -1,
                           }

        self.training()

    def training(self):
        del self.df_train["date_time"]

        # this code needs some revision, only supposed to work for now

        X = self.df_train.drop(['position', 'srch_id'], axis=1)
        y = self.df_train['position'].values.reshape(X.shape[0], 1)

        preprocessed_X = PreProcess(self.df_train).run()
        X_train, X_validation, y_train, y_validation = train_test_split(preprocessed_X, y, test_size=0.2,
                                                                        random_state=42)

        X_train = X_train.drop(['position', 'srch_id'], axis=1)
        X_validation = X_validation.drop(['position', 'srch_id'], axis=1)

        X_train = self.normalize(X_train)
        X_validation = self.normalize(X_validation)

        model = lightgbm.LGBMRegressor()

        y_train = pd.DataFrame(y_train)

        model.fit(X_train, y_train.values.ravel())
        print()
        print(model)
        expected_y = y_validation
        predicted_y = model.predict(X_validation)

        print("R2 Score:", round(metrics.r2_score(expected_y, predicted_y),2))
        print("MSE:", round(metrics.mean_squared_log_error(expected_y, predicted_y), 2))


    def normalize(self, df):
        sc = StandardScaler()
        sc.fit(df)
        df = sc.transform(df)

        return df


example_file = "data/shortened_data_5000.csv"
Light_GBM(example_file)
