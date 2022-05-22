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
from preprocessing import PreProcess


class Light_GBMRanker:
    """
    Light GBM Ranking model with sklearn API
    """

    def __init__(self, train_filepath, test_filepath):

        self.df_train_iterator = pd.read_csv(train_filepath, chunksize=100000)
        self.df_test_iterator = pd.read_csv(test_filepath, chunksize=100000)

        self.qids_train = None
        self.X_train = None
        self.y_train = np.array([])
        self.qids_validation = None
        self.X_validation = None
        self.y_validation = np.array([])

        self.all_ndcg = []

        self.model = lightgbm.LGBMRanker()

        self.LGBMRanker()

    @staticmethod
    def train_model_preprocessing(chunk_dataframe):

        train_df = chunk_dataframe[:800]
        validation_df = chunk_dataframe[800:]

        qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        y_train = train_df["ranking"]

        qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_validation = validation_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        y_validation = validation_df["ranking"]

        return qids_train, X_train, y_train, qids_validation, X_validation, y_validation

    def LGBMRanker(self):

        for training_chunk in enumerate(self.df_train_iterator):

            print(f'TRAINING DATA PREPROCESSING —— TRAINING CHUNK: {training_chunk[0]}')

            self.qids_train, self.X_train, self.y_train, self.qids_validation, self.X_validation, self.y_validation = \
                self.train_model_preprocessing(training_chunk[1])

            print('done.')

            if exists('lightGBM_model/lgbm_ranker/lgb_ranker.txt'):

                self.model = lightgbm.LGBMRanker(
                    boosting_type='dart',
                    objective="lambdarank",
                    metric='ndcg',
                    label_gain=[i for i in range(max(self.y_train.max(), self.y_validation.max()) + 1)],
                    learning_rate=0.0001,
                )

                gbm = self.model.fit(X=self.X_train,
                                     y=self.y_train,
                                     group=self.qids_train,
                                     eval_set=[(self.X_validation, self.y_validation)],
                                     eval_group=[self.qids_validation],
                                     eval_at=5,
                                     verbose=10,
                                     init_model='lightGBM_model/lgbm_ranker/lgb_ranker.txt',
                                     )
                gbm.booster_.save_model('lightGBM_model/lgbm_ranker/lgb_ranker.txt',
                                        num_iteration=gbm.best_iteration_)
                print(f"GBM: Saving iteration {training_chunk[0]} —— done.")

            else:

                self.model = lightgbm.LGBMRanker(
                    boosting_type='dart',
                    objective="lambdarank",
                    metric="ndcg",
                    label_gain=[i for i in range(max(self.y_train.max(), self.y_validation.max()) + 1)],
                    learning_rate=0.0001
                )

                gbm_init = self.model.fit(X=self.X_train,
                                          y=self.y_train,
                                          group=self.qids_train,
                                          eval_set=[(self.X_validation, self.y_validation)],
                                          eval_group=[self.qids_validation],
                                          eval_at=5,
                                          verbose=10,
                                          )
                gbm_init.booster_.save_model('lightGBM_model/lgbm_ranker/lgb_ranker.txt',
                                             num_iteration=gbm_init.best_iteration_)
                print(f"GBM_init: saving iteration == {training_chunk[0]}, done.")

            ####################################
            # PREDICTING THE PROPERTY LISTINGS #
            ####################################

        for pred_chunk in enumerate(self.df_test_iterator):

            print(f'TESTING DATA PREDICTIONS —— TESTING CHUNK: {pred_chunk[0]}')

            X_test = pred_chunk[1]
            final_predictions_df = pd.DataFrame(columns=['srch_id', 'prop_id'])
            sorted_srchs = sorted(X_test['srch_id'].unique())

            for srch_id in enumerate(sorted_srchs):
                X_test_per_site = X_test[X_test['srch_id'] == srch_id[1]]
                X_test_copy = X_test_per_site.copy()
                X_test_per_site = X_test_per_site.drop(['srch_id', 'prop_id'], axis=1)

                test_pred = self.model.predict(X_test_per_site)
                X_test_copy['ranking'] = test_pred

                X_test_copy = X_test_copy.sort_values(by=['ranking'], ascending=False)
                short_df = X_test_copy[['srch_id', 'prop_id']].copy()
                final_predictions_df = pd.concat([final_predictions_df, short_df], ignore_index=True)

            mode = 'w' if pred_chunk[0] == 0 else 'a'
            header = pred_chunk[0] == 0

            final_predictions_df.to_csv(r'lightGBM_model/lgbm_ranker/predictions_ranker.csv', index=False,
                                        header=header,
                                        mode=mode)
            print(f"——— successfully saved chunk{pred_chunk[0]} predicitons ——— ")

    def eval_ndcg(self, y_true, y_pred):

        eval_score = ndcg_score(y_true, y_pred)
        self.all_ndcg.append(eval_score)

        return ['weighted_ndcg', eval_score, True]


"""if exists('data/preprocessed/training_VU_DM.csv') and exists('data/preprocessed/testing_VU_DM.csv'):

    training_filepath = "data/preprocessed/training_VU_DM.csv"
    testing_filepath = "data/preprocessed/testing_VU_DM.csv"
    run_large_file_LGBM = Light_GBMRanker(training_filepath, testing_filepath)

else:
    unprocessed_training_filepath = "../original_data/training_set_VU_DM.csv"  # these file paths will differ from yours
    unprocessed_testing_filepath = "../original_data/test_set_VU_DM.csv"  # these file paths will differ from yours
    PreProcess(unprocessed_training_filepath, unprocessed_testing_filepath)

    processed_training_filepath = "data/preprocessed/training_VU_DM.csv"
    processed_testing_filepath = "data/preprocessed/testing_VU_DM.csv"
    run_large_file_LGBM = Light_GBMRanker(processed_training_filepath, processed_testing_filepath)"""


class small_data_LGBMRanker:

    def __init__(self, train_filepath, param_grid, test_filepath=None):

        self.df_train = pd.read_csv(train_filepath)
        self.df_test = pd.read_csv(test_filepath)
        self.all_ndcg = []

        self.param_grid = param_grid
        self.cv = StratifiedKFold(n_splits=3, shuffle=True)
        self.ensemble_df = pd.DataFrame()

    def example_train_preprocessing(self):

        train_df = self.df_train[:800]
        validation_df = self.df_train[800:]

        qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        y_train = train_df["ranking"]

        qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_validation = validation_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        y_validation = validation_df["ranking"]

        return qids_train, X_train, y_train, qids_validation, X_validation, y_validation

    def lgbm_ranker(self):
        qids_train, X_train, y_train, qids_validation, X_validation, y_validation, = self.example_train_preprocessing()

        model = lightgbm.LGBMRanker(boosting_type='dart',
                                    objective="lambdarank",
                                    metric='ndcg',
                                    ndcg_at=5,
                                    label_gain=[i for i in range(max(y_train.max(), y_validation.max()) + 1)],
                                    learning_rate=0.0001,
                                    )

        model.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(X_validation, y_validation)],
            eval_group=[qids_validation],
            eval_at=5,
            verbose=-1,
        )

        self.evaluate(model, 'LGBM_Ranker', True)

    def log_regression(self, ensemble):
        qids_train, X_train, y_train, qids_validation, X_validation, y_validation, = self.example_train_preprocessing()
        model = LogisticRegression(solver='sag', max_iter=10000)
        model.fit(X_train, y_train)
        if ensemble:
            return model
        else:
            self.evaluate(model, 'Logistic_Regression', False)

    def lgbm_classifier(self, ensemble):
        qids_train, X_train, y_train, qids_validation, X_validation, y_validation, = self.example_train_preprocessing()

        model = lightgbm.LGBMClassifier(boosting_type='dart',
                                        label_gain=[i for i in range(max(y_train.max(), y_validation.max()) + 1)])
        model.fit(X_train, y_train)
        if ensemble:
            return model
        else:
            self.evaluate(model, 'LGBM_classifier', False)

    def knn_classifier(self, ensemble):
        qids_train, X_train, y_train, qids_validation, X_validation, y_validation, = self.example_train_preprocessing()

        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(X_train, y_train)
        if ensemble:
            return model
        else:
            self.evaluate(model, 'KNN', False)

    def random_forest(self, ensemble):
        qids_train, X_train, y_train, qids_validation, X_validation, y_validation, = self.example_train_preprocessing()

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        if ensemble:
            return model
        else:
            self.evaluate(model, 'Random_Forest', False)

    def ensemble(self):
        lgbm = self.lgbm_classifier(True)
        knn = self.knn_classifier(True)
        forest = self.random_forest(True)
        logreg = self.log_regression(True)
        self.evaluate_ensemble([lgbm, knn, forest, logreg], ['lgbm', 'knn', 'forest', 'logreg'], 'median')

    def evaluate_ensemble(self, models, modelnames, eval_method):
        """
        eval_method: either mean or median
        """
        sorted_srchs = sorted(self.df_test['srch_id'].unique())
        searches_ranked = []
        for srch_id in enumerate(sorted_srchs):
            X_test_per_site = self.df_test[self.df_test['srch_id'] == srch_id[1]]
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
        final_predictions_df.to_csv(r'data/test_data_2500_predictions.csv', index=False, header=True)

        true_vals = pd.read_csv('data/2500/gold_data_2500.csv')
        sm = difflib.SequenceMatcher(None, final_predictions_df['prop_id'], true_vals['prop_id'])
        print("Similarity score on test set for ensemble model is: " + str(sm.ratio()))

    def evaluate(self, model, modelname, ranker):
        final_predictions_df = pd.DataFrame(columns=['srch_id', 'prop_id'])
        sorted_srchs = sorted(self.df_test['srch_id'].unique())

        for srch_id in enumerate(sorted_srchs):
            X_test_per_site = self.df_test[self.df_test['srch_id'] == srch_id[1]]
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
        final_predictions_df.to_csv(r'data/test_data_2500_predictions.csv', index=False, header=True)

        true_vals = pd.read_csv('data/2500/gold_data_2500.csv')
        sm = difflib.SequenceMatcher(None, final_predictions_df['prop_id'], true_vals['prop_id'])
        print("Similarity score on test set for {} is: ".format(modelname) + str(sm.ratio()))

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

lgbm = small_data_LGBMRanker(processed_training_filepath, param_grid, processed_testing_filepath)

#lgbm.lgbm_ranker()
"""lgbm.log_regression(False)
lgbm.lgbm_classifier(False)
lgbm.knn_classifier(False)
lgbm.random_forest(False)"""
lgbm.ensemble()
