from preprocessing import PreProcess
import numpy as np
import pandas as pd
import lightgbm
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from preprocessing import PreProcess
from sklearn.preprocessing import StandardScaler


class LGBM_affinity:

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
        self.testing()


    
    def training(self):
        del self.df_train["date_time"]
        self.clean_train = self.affinity_clean(self.df_train)

        variables = [i for i in self.clean_train.columns if i not in ["srch_query_affinity_score", "srch_id"]]
        X = self.clean_train.loc[:, variables]
        Y = self.clean_train.loc[:, "srch_query_affinity_score"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size = 0.2, random_state=42)

        # train_data = lightgbm.Dataset(X_train, label = y_train)
        # val_data = lightgbm.Dataset(X_val, label = y_val)

        model = lightgbm.LGBMRegressor()

        y_train = pd.DataFrame(y_train)

        model.fit(X_train, y_train.values.ravel())
        print()
        print(model)
        expected_y = y_val
        predicted_y = model.predict(X_val)

        print("--Training--")
        print("R2 Score:", round(metrics.r2_score(expected_y, predicted_y),2))
        print("MSE:", round(metrics.mean_squared_error(expected_y, predicted_y), 2))

        # model = lightgbm.Regressor(self.parameters, train_data, valid_sets=val_data, num_boost_round=5000, early_stopping_rounds=50)

        # y_train_pred = model.predict(X_train)
        # y_val_pred = model.predict(X_val)

        #return roc_auc_score(y_train, y_train_pred), roc_auc_score(y_val, y_val_pred)
    
    def testing(self):
        model = lightgbm.LGBMRegressor()
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        y_train = pd.DataFrame(y_train)

        model.fit(self.X_train, y_train.values.ravel())
        print()
        print(model)
        expected_y = y_test
        predicted_y = model.predict(X_test)

        print("--Testing--")
        print("R2 Score:", round(metrics.r2_score(expected_y, predicted_y),4))
        print("MSE:", round(metrics.mean_squared_error(expected_y, predicted_y), 4))

    def affinity_clean(self, df):
        affinity_clean_df = df[df["srch_query_affinity_score"].notna()]
        return affinity_clean_df


example_file = "/Users/omerkirmaz/Documents/VRIJE/Master/Year_1/P5/DMT/As2/Group118_Assn2/data/shortened_data_5000.csv"
LGBM_affinity(example_file)
# train_score, val_score = LGBM_affinity(example_file)
# print(f"Training performance:{train_score}\nValidation performance:{val_score}")


         