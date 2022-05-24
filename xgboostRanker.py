import pandas as pd
import xgboost as xgb
import sklearn
import numpy as np

from Preprocessing.preprocessing_XGB import preprocessing_1
from sklearn.datasets import *
from os.path import exists
from tqdm import tqdm
from ndcg import ndcg_score
import warnings

warnings.filterwarnings("ignore")


class Lambda_Ranker:
    """
    XGBoost — LamdaMart implementation
    """

    def __init__(self, TRAIN_DATA, TEST_DATA=None):

        self.df_train_iterator = pd.read_csv(TRAIN_DATA, chunksize=100000)
        self.df_test_iterator = pd.read_csv(TEST_DATA, chunksize=100000)

    @staticmethod
    def relevance(a):
        """
        updates the relevance score (dependent variable) for each row
        a: two rows (click_bool, book_bool) from a dataset
        returns: a value which is determined by the presence of either being booked, or clicked on
        """
        if a[0] == a[1] == 1:
            return 5
        elif a[0] == 1 and a[1] == 0:
            return 1
        else:
            return 0

    def dump_train_val_files(self, training_data, validation_data):
        """
        writes svmlight files in the format: RELEVANCE SCORE; QUERY_ID; FEATURES
        training_data: a dataframe (chunk) that serves as a training set
        validation_data: a dataframe (chunk) that serves as a validation set
        returns: nothing. just saves files in data/svmlight(...).txt
        """

        train_features = training_data.iloc[:, 1:-2].values
        train_relevance = training_data.iloc[:, -2:].apply(self.relevance, axis=1)

        validation_features = validation_data.iloc[:, 1:-2].values
        validation_relevance = validation_data.iloc[:, -2:].apply(self.relevance, axis=1)

        dump_svmlight_file(train_features, train_relevance, f'data/svmlight_training.txt',
                           query_id=training_data.srch_id)

        dump_svmlight_file(validation_features, validation_relevance, f'data/svmlight_validation.txt',
                           query_id=validation_data.srch_id)

    @staticmethod
    def dump_test_files(test_data):
        """
        same function as above, just saves a test file this time. this is a seperate function, since the processing
        of the test file is needed to be called in a later stage, subsequent of the training
        test_data: a dataframe (chunk) that serves as a testing set
        returns: nothing. just saves files in data/svmlight(...).txt
        """

        dump_svmlight_file(test_data.iloc[:, 1:-2].values, f'data/svmlight_test.txt', query_id=test_data.srch_id)

    @staticmethod
    def splitting_datasets(chunk_dataframe):
        """
        Splits the dataframe (chunk) into a training_set (95%), and validation_set (5%)
        chunk_dataframe: a dataframe chunk of the VU_DM_training.csv
        returns: qids_train, a list of counts of occurrences per qid; train_df: training_set; qids_validation:
        same as for train just for val; validation_df = validation set
        """

        tenth_df = len(chunk_dataframe) // 20

        train_df = chunk_dataframe.iloc[:tenth_df]
        validation_df = chunk_dataframe.iloc[tenth_df:]

        qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()

        return qids_train, train_df, qids_validation, validation_df

    def train_LambdaMART(self):
        """
        loads the XGBoost LamdaMart model per df chunk (if existent), updates the model,
        and saves the model under model/XGBRanker.json
        """

        for training_chunk in enumerate(self.df_train_iterator):

            print(f'TRAINING DATA PREPROCESSING —— TRAINING CHUNK: {training_chunk[0]}')

            # PREPROCESSING AND SAVING DATA TO XGBOOST READABLE FORMAT

            training_data = preprocessing_1(training_chunk[1])
            qids_train, X_train, qids_validation, X_validation = self.splitting_datasets(training_data)
            self.dump_train_val_files(X_train, X_validation)

            # LOADING THE SVMLIGHT FILES AND READING: RELEVANCE, FEATURES, AND QID'S

            xgb_training_data, xgb_training_labels = \
                sklearn.datasets.load_svmlight_file('data/svmlight_training.txt')
            xgb_validation_data, xgb_validation_labels = \
                sklearn.datasets.load_svmlight_file('data/svmlight_validation.txt')

            train_features = np.asarray(xgb_training_data.todense())
            validation_features = np.asarray(xgb_validation_data.todense())

            xgb_training_labels = np.array(xgb_training_labels).reshape(-1, 1)
            xgb_validation_labels = np.array(xgb_validation_labels).reshape(-1, 1)

            qids_validation = np.array(qids_validation).reshape(-1, 1)
            qids_train = np.array(qids_train).reshape(-1, 1)

            if exists("model/XGBRanker.json"):

                xgb_model = xgb.XGBRanker(booster='dart',
                                          objective='rank:ndcg',
                                          tree_method='hist',
                                          learning_rate=0.0001,
                                          )

                # self.xgb_model.load_model("model/XGBRanker.json")
                xgb_model.fit(train_features, xgb_training_labels,
                              xgb_model="model/XGBRanker.json",
                              group=qids_train,
                              # eval_set=[(validation_features, xgb_validation_labels)],
                              # eval_group=[qids_validation],
                              # eval_metric=['ndcg@5'],
                              # verbose=True
                              )
                xgb_model.save_model("model/XGBRanker.json")

                print(f"XGB: Saving iteration {training_chunk[0]} —— done.", '\n')

            else:

                xgb_model = xgb.XGBRanker(booster='dart',
                                          objective='rank:ndcg',
                                          tree_method='hist',
                                          learning_rate=0.0001,
                                          )
                xgb_model.fit(train_features, xgb_training_labels,
                              group=qids_train,
                              # eval_set=[(validation_features, xgb_validation_labels)],
                              # eval_group=[qids_validation],
                              # eval_metric=['ndcg@5'],
                              # verbose=True
                              )
                xgb_model.save_model("model/XGBRanker.json")

                print(f"XGB_init: saving initial iteration == {training_chunk[0]}, done.", '\n')
        print("TRAINING IS FINISHED —— PREDICTIONS INITIALIZING...")

    def predict(self):
        """
        loads the XGBRanker LamdaMart model from model/XGBRanker.json and predicts the scores for every search_id query
        separately. then it writes all predictions to the file model/predictions_ranker.csv
        """

        for pred_chunk in enumerate(self.df_test_iterator):

            print(f'TESTING DATA PREDICTIONS —— TESTING CHUNK: {pred_chunk[0]}')
            X_test = preprocessing_1(pred_chunk[1], type=0)

            model = xgb.XGBRanker()
            model.load_model("model/XGBRanker.json")

            final_predictions_df = pd.DataFrame(columns=['srch_id', 'prop_id'])
            sorted_srchs = sorted(X_test['srch_id'].unique())

            for srch_id in tqdm(sorted_srchs):
                X_test_per_site = X_test[X_test['srch_id'] == srch_id]
                X_test_copy = X_test_per_site.copy()
                X_test_per_site = X_test_per_site.drop(['srch_id'], axis=1)

                test_pred = model.predict(X_test_per_site)
                X_test_copy['ranking'] = test_pred

                # NOT QUITE SURE IF ITS SORTED IN THE CORRECT ORDER NOW

                X_test_copy = X_test_copy.sort_values(by=['ranking'], ascending=False)
                short_df = X_test_copy[['srch_id', 'prop_id']].copy()
                final_predictions_df = pd.concat([final_predictions_df, short_df], ignore_index=True)

            mode = 'w' if pred_chunk[0] == 0 else 'a'
            header = pred_chunk[0] == 0

            # final_predictions_df = final_predictions_df.sort_values(by=['srch_id'], ascending=True)
            # final_predictions_df.rename(columns={'property_id': 'prop_id'}, inplace=True)

            final_predictions_df.to_csv(r'model/predictions_ranker_eval.csv', index=False,
                                        header=header,
                                        mode=mode)
            self.print_fun_statement(pred_chunk[0])

    @staticmethod
    def print_fun_statement(enumerator):
        """
        just a print statement.
        """
        print(f"""
        
         —— successfully saved chunk {enumerator} predicitons —— 
        
        """)


# SHORT DATASET IN CASE THE LARGE DATASET IS OVERKILL
training_path = "data/training_set_VU_DM.csv"
testing_path = "data/test_set_VU_DM.csv"

# LARGE DATASETS EXPEDIA HOTEL PREDICTIONS
short_train = "data/5000/training_data_5000.csv"
short_test = "data/2500/testing_data_2500.csv"

# IF THE MODEL SHOULD BE EVALUATED, USE THESE FILEPATHS !!!
TRAIN_PATH = "data/training_set_VU_DM.csv"
TEST_PATH = "data/2500/testing_data_2500.csv"

if __name__ == "__main__":
    LamdaMART = Lambda_Ranker(training_path, testing_path)
    #LamdaMART.train_LambdaMART()
    LamdaMART.predict()

    evaluation = ndcg_score(true_filepath='data/2500/gold_data_2500.csv',
                            pred_filepath='model/predictions_ranker_eval.csv')
    print('nDCG score ==', evaluation)
