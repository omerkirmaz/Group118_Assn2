import statistics

import numpy as np
import pandas as pd
import lightgbm

from preprocessing import PreProcess
from os.path import exists
from sklearn.metrics import ndcg_score
from tqdm import tqdm


class Light_GBMRegressor:

    def __init__(self, train_filepath, test_filepath):

        train_cols = list(pd.read_csv(train_filepath, nrows=1))
        test_cols = list(pd.read_csv(test_filepath, nrows=1))

        self.df_train_iterator = pd.read_csv(train_filepath, usecols=[i for i in train_cols if i != 'date_time'],
                                             chunksize=100000)
        self.df_test_iterator = pd.read_csv(test_filepath, usecols=[i for i in test_cols if i != 'date_time'],
                                            chunksize=100000)

        self.qids_train = None
        self.X_train = None
        self.y_train = np.array([])
        self.qids_validation = None
        self.X_validation = None
        self.y_validation = np.array([])

        self.model = lightgbm.LGBMRegressor(boosting_type='dart',
                                            learning_rate=0.001)

        self.train_and_predict_LGBMModel()
        # self.only_predict()

    def train_model_preprocessing(self, chunk_dataframe):

        data_df = PreProcess(chunk_dataframe).run()
        train_df = data_df[:800]
        validation_df = data_df[800:]

        # qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", "ranking"], axis=1)
        y_train = train_df["ranking"].values.flatten()

        # qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_validation = validation_df.drop(["srch_id", "ranking"], axis=1)
        y_validation = validation_df["ranking"].values.flatten()

        return X_train, y_train, X_validation, y_validation

    def predictions_preprocess(self, prediction_dataframe):

        data_df = PreProcess(prediction_dataframe).run()

        return data_df

    def train_and_predict_LGBMModel(self):

        #  So with the regressor an issue i have is that the input datadoes not account for the queries. So therefore,
        #  i dont know how well the model will actually predict the ultimate score ...

        for training_chunk in enumerate(self.df_train_iterator):

            print(f'TRAINING DATA PREPROCESSING —— TRAINING CHUNK: {training_chunk[0]}')

            self.X_train, self.y_train, self.X_validation, self.y_validation = \
                self.train_model_preprocessing(training_chunk[1])

            print('done.')

            if exists('lightGBM_model/lgbm_regressor/lgbm_regressor.txt'):

                gbm = self.model.fit(self.X_train,
                                     self.y_train,
                                     eval_set=[(self.X_validation, self.y_validation)],
                                     init_model='lightGBM_model/lgbm_regressor/lgbm_regressor.txt'
                                     )
                print('ending trainign. now starting with testing..')
                gbm.booster_.save_model('lightGBM_model/lgbm_regressor/lgbm_regressor.txt',
                                        num_iteration=gbm.best_iteration_)
                print(f"GBM: Saving iteration {training_chunk[0]} —— done.")

            else:

                gbm_init = self.model.fit(X=self.X_train,
                                          y=self.y_train,
                                          eval_set=[(self.X_validation, self.y_validation)],
                                          )
                gbm_init.booster_.save_model('lightGBM_model/lgbm_regressor/lgbm_regressor.txt',
                                             num_iteration=gbm_init.best_iteration_)
                print(f"GBM_init: saving iteration == {training_chunk[0]}, done.")

            #  PREDICTING THE PROPERTY LISTINGS

        print("""LGBM-REGRESSOR WRITING PREDICTIONS:""")

        for pred_chunk in enumerate(self.df_test_iterator):

            print(f'TESTING DATA PREPROCESSING —— TESTING CHUNK: {pred_chunk[0]}')

            X_test = self.predictions_preprocess(pred_chunk[1])
            final_predictions_df = pd.DataFrame(columns=['srch_id', 'prop_id'])

            print('done.')

            for srch_id in enumerate(X_test['srch_id'].unique()):

                X_test_per_site = X_test[X_test['srch_id'] == srch_id[1]]
                X_test_copy = X_test_per_site.copy()
                X_test_per_site = X_test_per_site.drop(['srch_id', 'prop_id'], axis=1)

                test_pred = self.model.predict(X_test_per_site)
                X_test_copy['ranking'] = test_pred

                # NOT QUITE SURE IF ITS SORTED IN THE CORRECT ORDER NOW

                X_test_copy = X_test_copy.sort_values(by=['ranking'], ascending=False)
                short_df = X_test_copy[['srch_id', 'prop_id']].copy()
                final_predictions_df = pd.concat([final_predictions_df, short_df], ignore_index=True)

            mode = 'w' if pred_chunk[0] == 0 else 'a'
            header = pred_chunk[0] == 0

            final_predictions_df = final_predictions_df.sort_values(by=['srch_id'], ascending=True)
            final_predictions_df.to_csv(r'lightGBM_model/lgbm_regressor/predictions_regressor_new.csv', index=False,
                                        header=header,
                                        mode=mode)

            print(f"——— successfully saved chunk{pred_chunk[0]} predicitons ——— ")


training_filepath = "../original_data/training_set_VU_DM.csv"  # these filepaths will differ from yours
testing_filepath = "../original_data/test_set_VU_DM.csv"  # these filepaths will differ from yours
run_large_file_LGBM = Light_GBMRegressor(training_filepath, testing_filepath)


class small_data_LGBMRanker:

    def __init__(self, train_filepath, test_filepath=None, gold_path=None):

        train_cols = list(pd.read_csv(train_filepath, nrows=1))
        test_cols = list(pd.read_csv(test_filepath, nrows=1))

        self.df_train = pd.read_csv(train_filepath, usecols=[i for i in train_cols if i != 'date_time'])
        self.df_test = pd.read_csv(test_filepath, usecols=[i for i in test_cols if i != 'date_time'])
        self.gold = pd.read_csv(gold_path)

        self.LGBMRegressor_example()
        self.evaluate_regressor()

    def example_train_preprocessing(self):

        data_df = PreProcess(self.df_train).run()
        train_df = data_df[:800]
        validation_df = data_df[800:]

        # qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", 'ranking'], axis=1)
        y_train = train_df["ranking"]

        # qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_validation = validation_df.drop(["srch_id", 'ranking'], axis=1)
        y_validation = validation_df["ranking"]

        return X_train, y_train, X_validation, y_validation

    def example_predictions_preprocess(self):
        data_df = PreProcess(self.df_test).run()

        return data_df

    def LGBMRegressor_example(self):

        print('TRAINING DATA PREPROCESSING:')
        X_train, y_train, X_validation, y_validation, = self.example_train_preprocessing()
        print('done.')

        model = lightgbm.LGBMRegressor(boosting_type='dart',
                                       learning_rate=0.001)

        model.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=10,
        )

        final_predictions_df = pd.DataFrame(columns=['srch_id', 'prop_id'])

        print('TRAINING DATA PREPROCESSING:')
        X_test = self.example_predictions_preprocess()

        print('LGBM-REGRESSOR WRITING PREDICTIONS:')
        for srch_id in tqdm(X_test['srch_id'].unique()):
            X_test_per_site = X_test[X_test['srch_id'] == srch_id]
            X_test_copy = X_test_per_site.copy()
            X_test_per_site = X_test_per_site.drop(['srch_id'], axis=1)

            test_pred = model.predict(X_test_per_site)
            X_test_copy['position'] = test_pred

            # NOT QUITE SURE IF ITS SORTED IN THE CORRECT ORDER NOW

            X_test_copy = X_test_copy.sort_values(by=['position'], ascending=False)
            short_df = X_test_copy[['srch_id', 'prop_id']].copy()
            final_predictions_df = pd.concat([final_predictions_df, short_df], ignore_index=True)

        final_predictions_df = final_predictions_df.sort_values(by=['srch_id'], ascending=True)
        # final_predictions_df.rename(columns={'property_id': 'prop_id'}, inplace=True)
        final_predictions_df.to_csv(r'data/predictions/2500_predictions_regressor.csv', index=False, header=True)
        print('done.')

    def evaluate_regressor(self):

        pred = pd.read_csv('data/predictions/2500_predictions_regressor.csv')

        all_ndcg = []
        for srch_id in enumerate(pred['srch_id'].unique()):
            gold_df = pred[pred['srch_id'] == srch_id[1]]
            pred_df = self.gold[self.gold['srch_id'] == srch_id[1]]

            eval_score = ndcg_score(gold_df, pred_df)
            all_ndcg.append(eval_score)

        final_ndcg = statistics.mean(all_ndcg)

        print()
        print("—————— EVALUATION OF LGBM-REGRESSOR —————— ")
        print(f"\tNDCG-Score: {final_ndcg}")

# train_example_file = "data/5000/training_data_5000.csv"
# test_example_file = "data/2500/testing_data_2500.csv"
# gold_example_file = "data/2500/gold_data_2500.csv"
#
# small_data_LGBMRanker(train_example_file, test_example_file, gold_example_file)
