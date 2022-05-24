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

    def cut_columns(self, norm_df, delete_columns):
        for column in delete_columns:
            if column in norm_df.columns:
                del norm_df[column]

        return norm_df

    def fill_na(self, data_df):

        nan_columns = ["prop_review_score", "orig_destination_distance"]

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
        del_columns = [
            'date_time',
            'site_id',
            'visitor_location_country_id',
            'visitor_hist_adr_usd',
            'prop_country_id',
            'prop_brand_bool',
            'promotion_flag',
            'srch_destination_id',
            'random_bool',
            'srch_saturday_night_bool',
            'srch_query_affinity_score',
        ]

        for chunk in enumerate(self.df_train_iterator):
            print(f'PREPROCESSING CHUNK {chunk[0]}')
            train_df = chunk[1]
            for i in range(1, 9):
                rate = 'comp' + str(i) + '_rate'
                inv = 'comp' + str(i) + '_inv'
                diff = 'comp' + str(i) + '_rate_percent_diff'
                del_columns.extend([rate, inv, diff])
            train_df = self.create_dependent_column(train_df)
            train_df = self.cut_columns(train_df, del_columns)
            train_df = self.fill_na(data_df=train_df)

            mode = 'w' if chunk[0] == 0 else 'a'
            header = chunk[0] == 0

            train_df.to_csv(r'data/preprocessed/training_VU_DM.csv', index=False,
                            header=header,
                            mode=mode)

            print(f"——— successfully saved chunk {chunk[0]} ——— ")

    def save_test(self):
        del_columns = [
            'date_time',
            'site_id',
            'visitor_location_country_id',
            'visitor_hist_adr_usd',
            'prop_country_id',
            'prop_brand_bool',
            'promotion_flag',
            'srch_destination_id',
            'random_bool',
            'srch_saturday_night_bool',
            'srch_query_affinity_score',
        ]

        for chunk in enumerate(self.df_test_iterator):
            test_df = chunk[1]

            for i in range(1, 9):
                rate = 'comp' + str(i) + '_rate'
                inv = 'comp' + str(i) + '_inv'
                diff = 'comp' + str(i) + '_rate_percent_diff'
                del_columns.extend([rate, inv, diff])
            test_df = self.create_dependent_column(test_df)
            test_df = self.cut_columns(test_df, del_columns)
            test_df = self.fill_na(test_df)

            mode = 'w' if chunk[0] == 0 else 'a'
            header = chunk[0] == 0

            test_df.to_csv(r'data/preprocessed/testing_VU_DM.csv', index=False,
                           header=header,
                           mode=mode)
            print(f"——— successfully saved testing chunk {chunk[0]} predicitons ——— ")


# train_path = "data/5000/training_data_5000.csv"
# test_path = "data/2500/testing_data_2500.csv"
# PreProcess(train_path, test_path)
