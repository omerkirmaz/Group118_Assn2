import numpy as np
import pandas as pd
import lightgbm
from sklearn.metrics import ndcg_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GASearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_genetic.space import Continuous, Categorical, Integer
from os.path import exists
import difflib
import pickle
from preprocessing import PreProcess


class RankInstances:

    def __init__(self, train_filepath, param_grid, test_filepath=None):
        self.df_train_iterator = pd.read_csv(train_filepath, chunksize=100000)
        self.df_test_iterator = pd.read_csv(test_filepath, chunksize=100000)

        self.qids_train = None
        self.X_train = None
        self.y_train = []
        self.qids_validation = None
        self.X_validation = None
        self.y_validation = []

        self.all_ndcg = []

        self.param_grid = param_grid
        self.cv = StratifiedKFold(n_splits=3, shuffle=True)

        self.process_data()

    def process_data(self):
        """
        Process all chunks from the training data
        """
        for training_chunk in enumerate(self.df_train_iterator):

            print(f'TRAINING DATA PREPROCESSING —— TRAINING CHUNK: {training_chunk[0]}')

            self.train_model_preprocessing(training_chunk[1])

            print('done.')

    def train_model_preprocessing(self, chunk_dataframe):
        """
        Drop irrelevant columns from data and split the data
        :param chunk_dataframe: chunk of data to process
        """
        train_df = chunk_dataframe[:800]
        validation_df = chunk_dataframe[800:]

        self.qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        self.X_train = train_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        self.y_train = train_df["ranking"]

        self.qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        self.X_validation = validation_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        self.y_validation = validation_df["ranking"]

    def lgbm_ranker(self):
        """
        Make LGBM ranking model, fit it and evaluate performance
        """
        model = lightgbm.LGBMRanker(boosting_type='dart',
                                    objective="lambdarank",
                                    metric='ndcg',
                                    ndcg_at=5,
                                    label_gain=[i for i in range(max(max(self.y_train), max(self.y_validation)) + 1)],
                                    learning_rate=0.0001,
                                    )

        model.fit(
            X=self.X_train,
            y=self.y_train,
            group=self.qids_train,
            eval_set=[(self.X_validation, self.y_validation)],
            eval_group=[self.qids_validation],
            eval_at=5,
            verbose=-1,
        )
        pickle.dump(model, open('models/lgbm_ranker.sav', 'wb'))

        self.evaluate(model, 'LGBM_Ranker', True)

    def log_regression(self, ensemble):
        """
        Define classifier, train it, and evaluate it/ return it for further processing in the ensemble algorithm
        :param ensemble: whether the model will be combined into an ensemble model
        :return: model will be returned if it is to be combined into an ensemble model
        """
        model = LogisticRegression(solver='sag', max_iter=10000)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('models/log_regression.sav', 'wb'))
        if ensemble:
            return model
        else:
            self.evaluate(model, 'Logistic_Regression', False)

    def lgbm_classifier(self, ensemble):
        """
        Define classifier, train it, and evaluate it/ return it for further processing in the ensemble algorithm
        :param ensemble: whether the model will be combined into an ensemble model
        :return: model will be returned if it is to be combined into an ensemble model
        """
        model = lightgbm.LGBMClassifier(boosting_type='dart',
                                        label_gain=[i for i in range(max(max(self.y_train), max(self.y_validation)) + 1)])
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('models/lgbm_classifier.sav', 'wb'))
        if ensemble:
            return model
        else:
            self.evaluate(model, 'LGBM_classifier', False)

    def knn_classifier(self, ensemble):
        """
        Define classifier, train it, and evaluate it/ return it for further processing in the ensemble algorithm
        :param ensemble: whether the model will be combined into an ensemble model
        :return: model will be returned if it is to be combined into an ensemble model
        """
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('models/knn_classifier.sav', 'wb'))
        if ensemble:
            return model
        else:
            self.evaluate(model, 'KNN', False)

    def random_forest(self, ensemble):
        """
        Define classifier, train it, and evaluate it/ return it for further processing in the ensemble algorithm
        :param ensemble: whether the model will be combined into an ensemble model
        :return: model will be returned if it is to be combined into an ensemble model
        """
        model = RandomForestClassifier(n_estimators=100)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('models/random_forest.sav', 'wb'))
        if ensemble:
            return model
        else:
            self.evaluate(model, 'Random_Forest', False)

    def ensemble(self):
        """
        Obtain fitted models, and obtain the rankings for all of them.
        The resulting ordered dataframe is saved for later review
        """
        lgbm = self.lgbm_classifier(True)
        knn = self.knn_classifier(True)
        forest = self.random_forest(True)
        logreg = self.log_regression(True)
        for pred_chunk in enumerate(self.df_test_iterator):

            print(f'TESTING DATA PREDICTIONS —— TESTING CHUNK: {pred_chunk[0]}')
            predictions_df = self.evaluate_ensemble([lgbm, knn, forest, logreg], ['lgbm', 'knn', 'forest', 'logreg'],
                                                    'mean', pred_chunk[1])

            mode = 'w' if pred_chunk[0] == 0 else 'a'
            header = pred_chunk[0] == 0
            predictions_df.to_csv('predictions/ensemble/prediction.csv', index=False, header=header, mode=mode)
            print(f"——— successfully saved chunk {pred_chunk[0]} predictions ——— ")

    @staticmethod
    def evaluate_ensemble(models, modelnames, eval_method, df_test):
        """
        Evaluate a chunk of the test data using the selected models and combine the different rankings into one score
        :param models: list of the models to be used for the ensemble
        :param modelnames: the names of the respective models
        :param eval_method: Whether the mean or the median is used to calculate scores
        :param df_test: The test set to evaluate on
        :return: A sorted dataframe containing the search and property ID
        """
        sorted_srchs = sorted(df_test['srch_id'].unique())
        searches_ranked = []

        for srch_id in enumerate(sorted_srchs):
            X_test_per_site = df_test[df_test['srch_id'] == srch_id[1]]
            X_test_copy = X_test_per_site.copy()
            X_test_per_site = X_test_per_site.drop(['srch_id', 'prop_id'], axis=1)

            for i in range(len(models)):
                test_pred = models[i].predict_proba(X_test_per_site).tolist()
                column_name = 'ranking_{}'.format(modelnames[i])
                X_test_copy[column_name] = test_pred
                X_test_copy[['no_booking_prob_{}'.format(modelnames[i]),
                             'clicked_prob_{}'.format(modelnames[i]),
                             'booked_prob_{}'.format(modelnames[i])]] = pd.DataFrame(X_test_copy[column_name].tolist(),
                                                                                     index=X_test_copy.index)
            searches_ranked.append(X_test_copy)

        final_df = pd.concat(searches_ranked, ignore_index=True)
        final_df['mean_clicked'] = final_df.filter(like='clicked').mean(axis=1)
        final_df['median_clicked'] = final_df.filter(like='clicked').median(axis=1)

        final_df['mean_booked'] = final_df.filter(like='booked').mean(axis=1)
        final_df['median_booked'] = final_df.filter(like='booked').median(axis=1)

        final_df = final_df.sort_values(['{}_booked'.format(eval_method), '{}_clicked'.format(eval_method)], ascending=[False, False])

        final_predictions_df = final_df[['srch_id', 'prop_id']].copy()
        final_predictions_df = final_predictions_df.sort_values(by=['srch_id'], ascending=True)
        final_predictions_df.rename(columns={'property_id': 'prop_id'}, inplace=True)

        return final_predictions_df

    def evaluate(self, model, modelname, ranker):
        """
        Rank for all chunks of the test set using a provided model
        :param model: The model to evaluate
        :param modelname: The name of the model to evaluate
        :param ranker: If the lgbm_ranker is used, the evaluation has to account for the different method
        """
        for pred_chunk in enumerate(self.df_test_iterator):
            print(f'TESTING DATA PREDICTIONS —— TESTING CHUNK: {pred_chunk[0]}')
            X_test = pred_chunk[1]
            final_predictions_df = pd.DataFrame(columns=['srch_id', 'prop_id'])
            sorted_srchs = sorted(X_test['srch_id'].unique())

            for srch_id in enumerate(sorted_srchs):
                X_test_per_site = X_test[X_test['srch_id'] == srch_id[1]]
                X_test_copy = X_test_per_site.copy()
                X_test_per_site = X_test_per_site.drop(['srch_id', 'prop_id'], axis=1)

                if ranker:
                    test_pred = model.predict(X_test_per_site)
                    X_test_copy['ranking'] = test_pred
                    X_test_copy = X_test_copy.sort_values(['ranking'], ascending=False)
                else:
                    test_pred = model.predict_proba(X_test_per_site).tolist()
                    column_name = 'ranking_{}'.format(modelname)
                    X_test_copy[column_name] = test_pred
                    X_test_copy[['no_booking_prob', 'clicked_prob', 'booked_prob']] = pd.DataFrame(X_test_copy[column_name].tolist(), index=X_test_copy.index)
                    X_test_copy = X_test_copy.sort_values(['booked_prob', 'clicked_prob'], ascending=[False, False])

                short_df = X_test_copy[['srch_id', 'prop_id']].copy()
                final_predictions_df = pd.concat([final_predictions_df, short_df], ignore_index=True)

            final_predictions_df = final_predictions_df.sort_values(by=['srch_id'], ascending=True)
            final_predictions_df.rename(columns={'property_id': 'prop_id'}, inplace=True)

            mode = 'w' if pred_chunk[0] == 0 else 'a'
            header = pred_chunk[0] == 0

            final_predictions_df.to_csv('predictions/single_method/{}.csv'.format(modelname), index=False,
                                        header=header,
                                        mode=mode)
            print(f"——— successfully saved chunk {pred_chunk[0]} predicitons ——— ")

    def eval_ndcg(self, y_true, y_pred):

        eval_score = ndcg_score(y_true, y_pred, k=5)
        self.all_ndcg.append(eval_score)

        return ['weighted_ndcg', eval_score, True]


"""unprocessed_training_filepath = "data/5000/training_data_5000.csv"  # these file paths will differ from yours
        
unprocessed_testing_filepath = "data/2500/testing_data_2500.csv"  # these file paths will differ from yours
PreProcess(unprocessed_training_filepath, unprocessed_testing_filepath)"""

processed_training_filepath = "data/preprocessed/training_VU_DM.csv"
processed_testing_filepath = "data/preprocessed/testing_VU_DM.csv"

param_grid = {'learning_rate': Continuous(0.00001, 0.5, distribution='log-uniform'),
              'max_depth': Integer(2, 50),
              'num_leaves': Integer(2, 500),
              'feature_fraction': Continuous(0.01, 0.99),
              'bagging_fraction': Continuous(0.01, 0.99),
              'bagging_freq': Integer(0, 50),
              'max_bin': Integer(100, 500),
              'num_iterations': Integer(50, 200)
              }

lgbm = RankInstances(processed_training_filepath, param_grid, processed_testing_filepath)

#lgbm.lgbm_ranker()
"""lgbm.log_regression(False)
lgbm.lgbm_classifier(False)
lgbm.knn_classifier(False)"""
#lgbm.random_forest(False)
lgbm.ensemble()


# Calculate similarity score between predicted and gold data
true_vals = pd.read_csv('data/2500/gold_data_2500.csv')
predicted = pd.read_csv('predictions/single_method/random_forest.csv')
sm = difflib.SequenceMatcher(None, predicted['prop_id'], true_vals['prop_id'])
print("Similarity score on test set for ensemble is: " + str(sm.ratio()))
