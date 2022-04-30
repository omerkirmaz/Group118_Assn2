import pandas as pd
import random


class CuttingData:

    def __init__(self, filepath):

        self.df = pd.read_csv(filepath)
        self.shortened_df = pd.DataFrame(columns=self.df.columns.to_list())

        self.cutting()
        self.saving_csv()

    def cutting(self):
        total_num = self.df['srch_id'].unique()
        length_total = len(total_num)
        search_id_list = self.df['srch_id'].to_list()

        for i in range(round(int(length_total / 40))):
            column_id = random.choice(search_id_list)
            all_rows = self.df[self.df['srch_id'].values == column_id]
            self.shortened_df = pd.concat([self.shortened_df, all_rows], ignore_index=True)
            if i % 100 == 0:
                print(f"""{i} of {round((length_total / 40))} search ids appended to the dataframe
-----------------------------------------------------------------------------------------------------------------------
""")

    def saving_csv(self):
        self.shortened_df.to_csv(r'data/shortened_data_5000.csv', index=False, header=True)


csvfile_path = "data/training_set_VU_DM.csv"
to_cut = CuttingData(csvfile_path)
