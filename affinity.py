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

    def __init__(self, train_data, test_data, variables):
        self.df_train = pd.read_csv(train_data)
        self.df_test = pd.read_csv(test_data)

        del self.df_train["date_time"]
        del self.df_test["date_time"]

        self.clean_train = self.affinity_clean(self.df_train)
        self.clean_test = self.affinity_clean(self.df_test)

        self.X_train, self.X_val, self.y_train, self.y_val = [], [], [], []
        self.variables = variables

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

    def training(self):
        X = self.clean_train[variables]
        Y = self.clean_train["srch_query_affinity_score"]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        model = lightgbm.LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=300, device='gpu')

        y_train = pd.DataFrame(self.y_train)

        model.fit(self.X_train, self.y_train)
        print()
        print(model)
        predicted_y = model.predict(self.X_val)

        print("--Training--")
        print("R2 Score:", round(metrics.r2_score(self.y_val, predicted_y),2))
        print("MSE:", round(metrics.mean_squared_error(self.y_val, predicted_y), 2))

        return model
    
    def testing(self, model):
        X = self.clean_test[self.variables]
        Y = self.clean_test["srch_query_affinity_score"]

        predicted_y = model.predict(X)

        print("--Testing--")
        print("R2 Score:", round(metrics.r2_score(Y, predicted_y),4))
        print("MSE:", round(metrics.mean_squared_error(Y, predicted_y), 4))

    def affinity_clean(self, df):
        affinity_clean_df = df[df["srch_query_affinity_score"].notna()]
        return affinity_clean_df


train_data = "data/shortened_data_10000.csv"
test_data = "data/shortened_data_5000.csv"
variables = ['prop_review_score', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating',
             'price_usd', 'promotion_flag', 'srch_destination_id', 'prop_country_id']
lgbm = LGBM_affinity(train_data, test_data, variables)

model = lgbm.training()
lgbm.testing(model)

# train_score, val_score = LGBM_affinity(example_file)
# print(f"Training performance:{train_score}\nValidation performance:{val_score}")


         