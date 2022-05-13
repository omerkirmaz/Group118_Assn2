import statistics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class PreProcess:

    def __init__(self, dataframe):
        self.df = dataframe

        # self.run()
        # print(self.df.head(200).to_string())

        # print(self.df["price_usd"].head(200).to_string(), self.df["gross_bookings_usd"].head(200).to_string())
        # self.price_usd()

    def cut_columns(self):

        if 'date_time' in self.df.columns:
            del self.df['date_time']

        if 'click_bool' in self.df.columns:
            del self.df['click_bool']

        if 'booking_bool' in self.df.columns:
            del self.df['booking_bool']

        if 'gross_bookings_usd' in self.df.columns:
            del self.df['gross_bookings_usd']

        self.df.fillna(value=0, inplace=True)

        # all_cols = self.df.columns.tolist()
        #
        # for col in all_cols:
        #     missing = self.df[col].isnull().sum()
        #     if missing != 0:
        #         del self.df[col]

    def replace_nan_with_median(self):
        nan_columns = ["visitor_hist_starrating", "visitor_hist_adr_usd", "prop_review_score",
                       "orig_destination_distance"]

        for col in nan_columns:
            median = statistics.median(self.df[col].dropna())
            self.df[col].fillna(median, inplace=True)

    def normalize(self):

        if 'position' in self.df.columns:
            independent_df = self.df.drop(['position', 'srch_id'], axis=1)
            position_df = pd.DataFrame(self.df['position'].values.reshape(independent_df.shape[0], 1),
                                       columns=['position'])
            srch_id_df = pd.DataFrame(self.df['srch_id'].values.reshape(independent_df.shape[0], 1),
                                      columns=['srch_id'])

            scaler = MinMaxScaler()
            scaler.fit(independent_df)
            scaled = scaler.fit_transform(independent_df)
            scaled_df = pd.DataFrame(scaled, columns=independent_df.columns)

            rescaled_df = pd.concat([srch_id_df, scaled_df, position_df], axis=1)

        else:

            independent_df = self.df.drop(['srch_id'], axis=1)
            srch_id_df = pd.DataFrame(self.df['srch_id'].values.reshape(independent_df.shape[0], 1),
                                      columns=['srch_id'])
            prop_id_df = pd.DataFrame(self.df['prop_id'].values.reshape(independent_df.shape[0], 1),
                                      columns=['property_id'])
            prop_id_df['property_id'] = prop_id_df['property_id'].astype(int)

            scaler = MinMaxScaler()
            scaler.fit(independent_df)
            scaled = scaler.fit_transform(independent_df)
            scaled_df = pd.DataFrame(scaled, columns=independent_df.columns)

            rescaled_df = pd.concat([srch_id_df, prop_id_df, scaled_df], axis=1)

        return rescaled_df

    def run(self):
        self.replace_nan_with_median()
        self.cut_columns()
        normalized_df = self.normalize()

        return normalized_df

    # def price_usd(self):
    #
    #     visitor_location_df = self.df[self.df["site_id"].notna()]
    #     location_5 = visitor_location_df[visitor_location_df['site_id'] == 5]
    #
    #     #  print(visitor_location_df['visitor_location_country_id'].unique())
    #     #  print(visitor_location_df['site_id'].value_counts().to_string())
    #     #  print(property_country_df.head(200).to_string())
    #
    #     booked_location_5 = location_5[location_5['gross_bookings_usd'].notna()]
    #
    #     #  print(booked_location_5.head(200).to_string())
    #     #  print(booked_location_5['visitor_location_country_id'].unique())
    #     #  print(booked_location_5['visitor_location_country_id'].value_counts().to_string())
    #
    #     per_night_df = booked_location_5[booked_location_5['srch_length_of_stay'] == 1]
    #     per_night_df = booked_location_5[booked_location_5['promotion_flag'] == 0]
    #
    #     print("""
    #     PRICE PER NIGHT, TOTAL PRICE, THE PERCENTAGE OF DIFFERENCE BETWEEEN THEM"
    #     """)
    #
    #     for row in range(len(per_night_df['price_usd'])):
    #         price = per_night_df['price_usd'].loc[per_night_df.index[row]]
    #         stay = per_night_df['srch_length_of_stay'].loc[per_night_df.index[row]]
    #         total_booking = per_night_df['gross_bookings_usd'].loc[per_night_df.index[row]]
    #         diff = total_booking - price
    #         percent = round((diff / total_booking) * 100, 2)
    #
    #         print(price * stay, "== price per night |||", total_booking, "== total booking",
    #               "||| percent difference = ",
    #               percent)
    #         print("---------------------------------------------------------------------------------------------------")

# csv_df = pd.read_csv("data/shortened_data_5000.csv")
# preprocessed_csv = PreProcess(csv_df)