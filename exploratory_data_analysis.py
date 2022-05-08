import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ExploratoryDataAnalysis:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

        self.print_EDA()

    def check_null_columns(self):
        """
        function that checks if a column is just composed of null values
        :return: returns a list with each col name and a boolean
        """
        total = self.df.isnull().sum().sort_values(ascending=False)
        percent = (self.df.isnull().sum() / self.df['srch_id'].count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data.head(len(self.df.columns.tolist())).to_string())

    def visualise_and_save_correlations(self):
        half_df = len(self.df.columns.to_list()) // 2
        split_df = [self.df.iloc[:, : half_df], self.df.iloc[:, half_df:]]

        for i in enumerate(split_df):
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10)
            sns.heatmap(round(i[1].corr(), 2), cmap='coolwarm', ax=ax, annot=True, linewidths=2)
            fig.savefig(f"figures/correlations{i[0]}.png")
            fig.show()

    def print_EDA(self):
        print("""EXPLORATORY DATA ANALYSIS
        
        --------------------------------------------------------------------------------------------------------
        All the features of the dataset 
        --------------------------------------------------------------------------------------------------------
        
        """)
        print(self.df.info())

        print("""

        --------------------------------------------------------------------------------------------------------
        Percentage of all NaN values for each column        
        --------------------------------------------------------------------------------------------------------

        """)
        self.check_null_columns()

        print("""

        --------------------------------------------------------------------------------------------------------
        Heatmap of all correlations —–> check figures/correlations.png
        --------------------------------------------------------------------------------------------------------

        """)
        self.visualise_and_save_correlations()


train_set_filepath = "data/shortened_data_5000.csv"
to_preprocess = ExploratoryDataAnalysis(train_set_filepath)
