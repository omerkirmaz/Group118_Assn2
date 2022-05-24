import numpy as np
import pandas as pd
import lightgbm
import sklearn
from sklearn.metrics import ndcg_score, make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GroupKFold

from os.path import exists
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

        train_df = chunk_dataframe[:80000]
        validation_df = chunk_dataframe[80000:]

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
                                     categorical_feature=['visitor_location_country_id',
                                                          'prop_country_id',
                                                          'prop_brand_bool',
                                                          'promotion_flag',
                                                          'srch_destination_id',
                                                          ],
                                     init_model='lightGBM_model/lgbm_ranker/lgb_ranker.txt'
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
                                          categorical_feature=['visitor_location_country_id',
                                                               'prop_country_id',
                                                               'prop_brand_bool',
                                                               'promotion_flag',
                                                               'srch_destination_id'
                                                               ]
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
            print(f"——— successfully saved chunk{pred_chunk[0]} predictions ——— ")

    def eval_ndcg(self, y_true, y_pred):

        eval_score = ndcg_score(y_true, y_pred)
        self.all_ndcg.append(eval_score)

        return ['weighted_ndcg', eval_score, True]


"""if exists('data/preprocessed/training_VU_DM.csv') and exists('data/preprocessed/testing_VU_DM.csv'):

    training_filepath = "data/preprocessed/training_VU_DM.csv"
    testing_filepath = "data/preprocessed/testing_VU_DM.csv"
    run_large_file_LGBM = Light_GBMRanker(training_filepath, testing_filepath)

else:
    unprocessed_training_filepath = "data/full_data/training_set_VU_DM.csv"  # these file paths will differ from yours
    unprocessed_testing_filepath = "data/full_data/test_set_VU_DM.csv"  # these file paths will differ from yours
    PreProcess(unprocessed_training_filepath, unprocessed_testing_filepath)

    #processed_training_filepath = "data/preprocessed/training_VU_DM.csv"
    #processed_testing_filepath = "data/preprocessed/testing_VU_DM.csv"
    #run_large_file_LGBM = Light_GBMRanker(processed_training_filepath, processed_testing_filepath)"""


class small_data_LGBMRanker:

    def __init__(self, train_filepath, test_filepath=None):

        self.df_train = pd.read_csv(train_filepath)
        self.df_test = pd.read_csv(test_filepath)
        self.all_ndcg = []
        self.param_grid = {'learning_rate': [0.001, 0.0001, 0.00001],
                           'num_leaves': [31, 51, 81],
                           'feature_fraction': [0.8, 0.9, 1.0],
                           'max_bin': [255, 355, 455],
                           'num_iterations': [100, 200, 300],
                           'min_data_in_leaf': [20, 40, 60],
                           'lambda_l1': [0, 1, 1.5]
                           }

        self.LGBMRanker_example()

    def example_train_preprocessing(self):

        train_df = self.df_train[:4500]
        validation_df = self.df_train[4500:]

        qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        y_train = train_df["ranking"]

        qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_validation = validation_df.drop(["srch_id", 'ranking', 'prop_id'], axis=1)
        y_validation = validation_df["ranking"]

        cv = GroupKFold(n_splits=10)

        return qids_train, X_train, y_train, qids_validation, X_validation, y_validation, cv

    def LGBMRanker_example(self):
        qids_train, X_train, y_train, qids_validation, X_validation, y_validation, cv = \
            self.example_train_preprocessing()

        X_test = self.df_test
        model = lightgbm.LGBMRanker(boosting_type='dart',
                                    objective="lambdarank",
                                    metric='ndcg',
                                    ndcg_at=5,
                                    label_gain=[i for i in range(max(y_train.max(), y_validation.max()) + 1)],
                                    learning_rate=0.0001,
                                    random_state=42
                                    )
        gsearch = HalvingGridSearchCV(estimator=model, param_grid=self.param_grid, cv=cv, scoring=ndcg_score)
        gsearch.fit(X=X_train, y=y_train, verbose=10)

        print(gsearch.best_params_, gsearch.best_score_)

        """model.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(X_validation, y_validation)],
            eval_group=[qids_validation],
            #eval_metric=self.eval_ndcg,
            eval_at=5,
            #ndcg_at=5,
            verbose=10
            # xe_ndcg_mart
        )

        final_predictions_df = pd.DataFrame(columns=['srch_id', 'prop_id'])
        sorted_srchs = sorted(X_test['srch_id'].unique())

        for srch_id in enumerate(sorted_srchs):
            X_test_per_site = X_test[X_test['srch_id'] == srch_id[1]]
            X_test_copy = X_test_per_site.copy()
            X_test_per_site = X_test_per_site.drop(['srch_id', 'prop_id'], axis=1)

            test_pred = model.predict(X_test_per_site)
            X_test_copy['ranking'] = test_pred

            X_test_copy = X_test_copy.sort_values(by=['ranking'], ascending=False)
            short_df = X_test_copy[['srch_id', 'prop_id']].copy()
            final_predictions_df = pd.concat([final_predictions_df, short_df], ignore_index=True)

        final_predictions_df = final_predictions_df.sort_values(by=['srch_id'], ascending=True)
        final_predictions_df.rename(columns={'property_id': 'prop_id'}, inplace=True)
        final_predictions_df.to_csv(r'data/test_data_5000_predictions.csv', index=False, header=True)"""

    def eval_ndcg(self, y_true, y_pred):

        eval_score = ndcg_score(y_true, y_pred, k=5)
        self.all_ndcg.append(eval_score)

        return ['weighted_ndcg', eval_score, True]


"""unprocessed_training_filepath = "data/5000/training_data_5000.csv"  # these file paths will differ from yours
unprocessed_testing_filepath = "data/2500/testing_data_2500.csv"  # these file paths will differ from yours
PreProcess(unprocessed_training_filepath, unprocessed_testing_filepath)"""

train_example_file = "data/preprocessed/training_VU_DM.csv"
test_example_file = "data/preprocessed/testing_VU_DM.csv"

small_data_LGBMRanker(train_example_file, test_example_file)
