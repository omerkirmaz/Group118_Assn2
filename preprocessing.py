import pandas as pd

from tqdm import tqdm


class PreProcess:

    def __init__(self, train_filepath, test_filepath):

        train_cols = list(pd.read_csv(train_filepath, nrows=1))
        test_cols = list(pd.read_csv(test_filepath, nrows=1))

        self.df_train_iterator = pd.read_csv(train_filepath, usecols=[i for i in train_cols if i != 'date_time'],
                                             chunksize=100000)
        self.df_test_iterator = pd.read_csv(test_filepath, usecols=[i for i in test_cols if i != 'date_time'],
                                            chunksize=100000)

        self.save_train()
        self.save_test()

    def cut_columns(self, norm_df):

        if 'date_time' in norm_df.columns:
            del norm_df['date_time']

        if 'click_bool' in norm_df.columns:
            del norm_df['click_bool']

        if 'booking_bool' in norm_df.columns:
            del norm_df['booking_bool']

        if 'gross_bookings_usd' in norm_df.columns:
            del norm_df['gross_bookings_usd']

        return norm_df

    def replace_nan_with_median(self, data_df):

        nan_columns = ["visitor_hist_starrating", "visitor_hist_adr_usd", "prop_review_score",
                       "orig_destination_distance"]

        for col in nan_columns:
            median = data_df[col].median()
            data_df[col].fillna(median, inplace=True)

        data_df.fillna(value=0, inplace=True)
        return data_df

    def create_dependent_column(self, data_df):

        if 'position' in data_df:

            normalized_df = pd.DataFrame()

            for search_id in tqdm(data_df['srch_id'].unique()):
                search_id_df = data_df[data_df['srch_id'] == search_id]

                booked_df = search_id_df[search_id_df['booking_bool'] == 1]
                booked_list = [5] * len(booked_df.index)

                clicked_on_df = search_id_df[(search_id_df['click_bool']) == 1 & (search_id_df['booking_bool'] == 0)]
                clicked_on_list = [1] * len(clicked_on_df.index)

                position_df = search_id_df[(search_id_df['booking_bool'] == 0) &
                                           (search_id_df['click_bool'] == 0)]
                pos_list = [0] * len(position_df.index)

                ranking_list = booked_list + clicked_on_list + pos_list

                search_id_df = pd.concat([booked_df, clicked_on_df, position_df])

                search_id_df['ranking'] = ranking_list
                search_id_df = search_id_df.drop_duplicates()
                search_id_df = search_id_df.drop(['gross_bookings_usd', 'click_bool', 'booking_bool', 'position'],
                                                 axis=1)

                normalized_df = pd.concat([normalized_df, search_id_df], ignore_index=True)

            return normalized_df

        else:
            normalized_df = data_df.copy()

            return normalized_df

    def save_train(self):

        for chunk in enumerate(self.df_train_iterator):
            print(f'PREPROCESSING CHUNK {chunk[0]}')
            train_df = chunk[1]

            train_df = self.replace_nan_with_median(data_df=train_df)
            train_df = self.create_dependent_column(train_df)

            mode = 'w' if chunk[0] == 0 else 'a'
            header = chunk[0] == 0

            train_df.to_csv(r'data/preprocessed/training_VU_DM.csv', index=False,
                            header=header,
                            mode=mode)

            print(f"——— successfully saved chunk {chunk[0]} ——— ")

    def save_test(self):

        for chunk in enumerate(self.df_test_iterator):
            test_df = self.replace_nan_with_median(chunk[1])
            test_df = self.create_dependent_column(test_df)

            mode = 'w' if chunk[0] == 0 else 'a'
            header = chunk[0] == 0

            test_df.to_csv(r'data/preprocessed/testing_VU_DM.csv', index=False,
                           header=header,
                           mode=mode)
            print(f"——— successfully saved testing chunk {chunk[0]} predicitons ——— ")


# train_path = "data/5000/training_data_5000.csv"
# test_path = "data/2500/testing_data_2500.csv"
# PreProcess(train_path, test_path)
