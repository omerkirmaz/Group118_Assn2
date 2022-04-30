import pandas as pd


class ExploratoryDataAnalysis:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

        self.reformat_prop_desirability()
        # print(self.df['prop_location_score'].head(30).to_list())

    def check_null_columns(self):
        """
        function that checks if a column is just composed of null values
        :return: returns a list with each col name and a boolean
        """
        col_null_list = []

        for col in self.df.columns:
            bool_value = self.df[col].isnull().values.all()
            col_null_list.append((col, bool_value))

        return col_null_list

    def reformat_prop_desirability(self):
        """
        normalises and takes the mean of the two property desirbality scores and prints the correlation to click
        :return: print statements and an updated df with mean values of normalised property desirability
        """

        col_names = ['prop_location_score1', 'prop_location_score2']

        for col in col_names:
            minimum = self.df[col].min()
            maximum = self.df[col].max()
            self.df[col] = (round((self.df[col] - minimum) / (maximum - minimum), 4))

        self.df['prop_location_score'] = round(((self.df['prop_location_score1'] +
                                                 self.df['prop_location_score2']) / 2), 4)
        self.print_reformat_prop_desirability()

        self.df.drop(columns=['prop_location_score1', 'prop_location_score2'])

    def print_reformat_prop_desirability(self):

        print(f'Correlation Click -- Prop Score1 == {self.df["click_bool"].corr(self.df["prop_location_score1"])}')
        print(f'------------------------------------------------------------------------------------------------------')
        print(f'Correlation Click -- Prop Score2 == {self.df["click_bool"].corr(self.df["prop_location_score2"])}')
        print(f'------------------------------------------------------------------------------------------------------')
        print(f'Correlation Click -- Prop Score Mean == {self.df["click_bool"].corr(self.df["prop_location_score"])}')


train_set_filepath = "data/shortened_data_5000.csv"
to_preprocess = ExploratoryDataAnalysis(train_set_filepath)
