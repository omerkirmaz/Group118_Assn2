import pprint
import statistics

import pandas as pd


class PreProcess:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

        self.replace_nan_with_median()
        # print(self.df["price_usd"].head(200).to_string(), self.df["gross_bookings_usd"].head(200).to_string())
        # self.price_per_night()
        self.price_usd()

    def replace_nan_with_median(self):
        nan_columns = ["visitor_hist_starrating", "visitor_hist_adr_usd", "prop_review_score",
                       "orig_destination_distance"]

        for col in nan_columns:
            median = statistics.median(self.df[col].dropna())
            self.df[col].fillna(median, inplace=True)

    def price_usd(self):

        visitor_location_df = self.df[self.df["site_id"].notna()]
        location_5 = visitor_location_df[visitor_location_df['site_id'] == 5]

        #  print(visitor_location_df['visitor_location_country_id'].unique())
        #  print(visitor_location_df['site_id'].value_counts().to_string())
        #  print(property_country_df.head(200).to_string())

        booked_location_5 = location_5[location_5['gross_bookings_usd'].notna()]

        #  print(booked_location_5.head(200).to_string())
        #  print(booked_location_5['visitor_location_country_id'].unique())
        #  print(booked_location_5['visitor_location_country_id'].value_counts().to_string())

        per_night_df = booked_location_5[booked_location_5['srch_length_of_stay'] == 1]
        per_night_df = booked_location_5[booked_location_5['promotion_flag'] == 0]

        print("""
        PRICE PER NIGHT, TOTAL PRICE, THE PERCENTAGE OF DIFFERENCE BETWEEEN THEM"
        """)

        for row in range(len(per_night_df['price_usd'])):
            price = per_night_df['price_usd'].loc[per_night_df.index[row]]
            stay = per_night_df['srch_length_of_stay'].loc[per_night_df.index[row]]
            total_booking = per_night_df['gross_bookings_usd'].loc[per_night_df.index[row]]
            diff = total_booking - price
            percent = round((diff / total_booking) * 100, 2)

            print(price * stay, "== price per night, ", total_booking, "== total booking", "|| percent difference = ",
                  percent)
            print("---------------------------------------------------------------------------------------------------")


csv_file = "data/shortened_data_5000.csv"
preprocessed_csv = PreProcess(csv_file)
