import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree, linear_model
from sklearn.neighbors import KNeighborsRegressor
from preprocessing import PreProcess
import math


class BasicClassifiers:
    def __init__(self, train_data, test_data, variables):
        self.df_train = pd.read_csv(train_data)
        self.df_test = pd.read_csv(test_data)

        del self.df_train["date_time"]
        del self.df_test["date_time"]

        self.clean_train = self.affinity_clean(self.df_train)
        self.clean_test = self.affinity_clean(self.df_test)

        processing_train = PreProcess(self.clean_train)
        self.clean_train = processing_train.run()

        processing_test = PreProcess(self.clean_test)
        self.clean_test = processing_test.run()

        self.X_train = self.clean_train[variables]
        self.Y_train = self.clean_train["srch_query_affinity_score"]

        self.X_test = self.clean_test[variables]
        self.Y_test = self.clean_test["srch_query_affinity_score"]

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.Y_train, test_size=0.2, random_state=42)

    def knn(self):
        knn_model = KNeighborsRegressor(n_neighbors=4)
        knn_model.fit(self.X_train, self.y_train)
        self.evaluate('KNN with K=4', knn_model)

    def tree_regression(self):
        tree_model = tree.DecisionTreeRegressor(max_depth=30)
        tree_model.fit(self.X_train, self.y_train)
        self.evaluate("Decision Tree Regressor", tree_model)

    def linear_regression(self):
        # Performs so bad that it shouldn't be included in the ensemble method
        lin_model = linear_model.LinearRegression()
        lin_model.fit(self.X_train, self.y_train)
        self.evaluate("Multiple Linear Regression", lin_model)

    def evaluate(self, modelname, model):
        val_y_pred = model.predict(self.X_val)
        test_y_pred = model.predict(self.X_test)

        print("-- Validation of {} --".format(modelname))
        print("R2 Score:", round(metrics.r2_score(self.y_val, val_y_pred), 2))
        print("RMSE:", round(math.sqrt(metrics.mean_squared_error(self.y_val, val_y_pred)), 2))
        print("MAE:", round(metrics.mean_absolute_error(self.y_val, val_y_pred), 2))
        print("-- Test of {} --".format(modelname))
        print("R2 Score:", round(metrics.r2_score(self.Y_test, test_y_pred), 2))
        print("RMSE:", round(math.sqrt(metrics.mean_squared_error(self.Y_test, test_y_pred)), 2))
        print("MAE:", round(metrics.mean_absolute_error(self.Y_test, test_y_pred), 2))
        print('\n')

    def affinity_clean(self, df):
        affinity_clean_df = df[df["srch_query_affinity_score"].notna()]
        return affinity_clean_df


train_data = "data/shortened_data_10000.csv"
test_data = "data/shortened_data_5000.csv"
variables = ['prop_review_score', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating',
             'price_usd', 'promotion_flag', 'srch_destination_id', 'prop_country_id']
classifiers = BasicClassifiers(train_data, test_data, variables)
classifiers.knn()
classifiers.tree_regression()
classifiers.linear_regression()
