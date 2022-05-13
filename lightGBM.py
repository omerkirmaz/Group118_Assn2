import pandas as pd
import lightgbm

from preprocessing import PreProcess
from os.path import exists


class Light_GBM:

    def __init__(self, train_filepath, test_filepath):

        self.df_train_iterator = pd.read_csv(train_filepath, chunksize=100000)
        self.df_test_iterator = pd.read_csv(test_filepath, chunksize=100000)

        self.training_LGBMRanker_chunks()
        self.lightGBM_predict()

    def train_model_preprocessing(self, chunk_dataframe):

        data_df = PreProcess(chunk_dataframe).run()
        train_df = data_df[:800]
        validation_df = data_df[800:]

        qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", "position"], axis=1)
        y_train = train_df["position"]

        qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", "position"], axis=1)
        y_train = train_df["position"]

        qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_validation = validation_df.drop(["srch_id", "position"], axis=1)
        y_validation = validation_df["position"]

        return qids_train, X_train, y_train, qids_validation, X_validation, y_validation

    def predictions_preprocess(self, prediction_dataframe):

        data_df = PreProcess(prediction_dataframe).run()

        qids_train = data_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_test = data_df.drop(["srch_id"], axis=1)

        return qids_train, X_test

    def training_LGBMRanker_chunks(self):

        for training_chunk in enumerate(self.df_train_iterator):

            qids_train, X_train, y_train, qids_validation, X_validation, y_validation = self.train_model_preprocessing(
                training_chunk[1])

            if exists('lightGBM_model/lgb_ranker.txt'):

                model = lightgbm.LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    label_gain=[i for i in range(max(y_train.max(), y_validation.max()) + 1)]

                )

                gbm = model.fit(X=X_train,
                                y=y_train,
                                group=qids_train,
                                eval_set=[(X_validation, y_validation)],
                                eval_group=[qids_validation],
                                eval_at=10,
                                verbose=10,
                                init_model='lightGBM_model/lgb_ranker.txt'
                                )
                gbm.booster_.save_model('lightGBM_model/lgb_ranker.txt', num_iteration=gbm.best_iteration_)
                print(f"GBM: Saving iteration {training_chunk[0]} —— done.")

            else:

                model = lightgbm.LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    label_gain=[i for i in range(max(y_train.max(), y_validation.max()) + 1)]
                )

                gbm_init = model.fit(X=X_train,
                                     y=y_train,
                                     group=qids_train,
                                     eval_set=[(X_validation, y_validation)],
                                     eval_group=[qids_validation],
                                     eval_at=100,
                                     verbose=100
                                     )
                gbm_init.booster_.save_model('lightGBM_model/lgb_ranker.txt', num_iteration=gbm_init.best_iteration_)
                print(f"GBM_init: saving iteration == {training_chunk[0]}, done.")

    def lightGBM_predict(self):

        for pred_chunk in enumerate(self.df_test_iterator):

            qids_train, X_test = PreProcess.run(pred_chunk[1])
            test_pred = lightgbm.LGBMRanker().predict(X_test)

            mode = 'w' if pred_chunk == 0 else 'a'
            header = pred_chunk == 0

            test_pred.to_csv(
                "predictions/dst_data.csv.gz",
                index=False,  # Skip index column
                header=header,
                mode=mode)
            print(f"——— successfully saved chunk{pred_chunk[0]} predicitons ——— ")


training_filepath = "../original_data/training_set_VU_DM.csv"  # these filepaths will differ from yours
testing_filepath = "../original_data/training_set_VU_DM.csv"  # these filepaths will differ from yours
run_large_file_LGBM = Light_GBM(training_filepath, testing_filepath)


class small_data_LGBMRanker:

    def __init__(self, train_filepath, test_filepath=None):
        self.df_train = pd.read_csv(train_filepath)
        self.df_test = pd.read_csv(test_filepath)

    def example_train_preprocessing(self):
        data_df = PreProcess(self.df_train).run()
        train_df = data_df[:800]
        validation_df = data_df[800:]

        qids_train = train_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_train = train_df.drop(["srch_id", "position"], axis=1)
        y_train = train_df["position"]

        qids_validation = validation_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_validation = validation_df.drop(["srch_id", "position"], axis=1)
        y_validation = validation_df["position"]

        return qids_train, X_train, y_train, qids_validation, X_validation, y_validation

    def example_predictions_preprocess(self):
        data_df = PreProcess(self.df_test).run()

        qids_train = data_df.groupby("srch_id")["srch_id"].count().to_numpy()
        X_test = data_df.drop(["srch_id"], axis=1)

        return qids_train, X_test

    def train_LGBMRanker_example(self):
        qids_train, X_train, y_train, qids_validation, X_validation, y_validation = self.example_train_preprocessing()

        model = lightgbm.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[i for i in range(max(y_train.max(), y_validation.max()) + 1)]

        )

        model.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(X_validation, y_validation)],
            eval_group=[qids_validation],
            eval_at=10,
            verbose=10,
            init_model='light_GBM_model/model.txt'
        )

    def example_lightGBM_predict(self):
        qids_train, X_test = self.example_predictions_preprocess()
        test_pred = lightgbm.LGBMRanker().predict(X_test)

        return test_pred

# example_file = "data/shortened_data_5000.csv"
# Light_GBM(example_file)
