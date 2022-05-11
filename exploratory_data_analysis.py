import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class ExploratoryDataAnalysis:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

        # self.save_search_affinity_score_dataframe()
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

    def save_search_affinity_score_dataframe(self):
        search_affinity_score_df = self.df[self.df['srch_query_affinity_score'].notna()]
        print(search_affinity_score_df.head(100).to_string())

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

        print("""

        --------------------------------------------------------------------------------------------------------
        Plots for columns —–> check figures/columns.png
        --------------------------------------------------------------------------------------------------------

        """)
        #self.plotting()

    def plotting(self):
        """
        This function plots a chart for every column. Some columns shouldnt be plotted though. It takes a long time. So
        instead, you might want to uncomment the code and add the columns which you want to plot to that list and
        update the for loop
        :return: plots saved in /figures directory
        """
        # wanted_cols = [""]

        for col in reversed(self.df.columns):
            fig, ax = plt.subplots()
            fig.set_size_inches(13, 8)
            sns.countplot(col, data=self.df, order=self.df[col].unique().sort(), ax=ax)
            fig.savefig(f"figures/{col}.png")
            # fig.show


train_set_filepath = "data/shortened_data_5000.csv"
to_preprocess = ExploratoryDataAnalysis(train_set_filepath)

