import pandas as pd


class DataProcessor:

    def __init__(self, data):
        self.data = data

    def std_filter(self, number):
        """
        Select the top 'number' columns based on their standard deviation.
        """
        index = list(self.data.std().sort_values(ascending=False)[0:number].index)
        return self.data[index]

    def MinmaxVARIABLES(self, number):
        """
        Normalize the data to be between 0 and 1.
        """
        data = self.std_filter(number)
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        return data

    def sort_corr(self, number):
        """
        Sort columns based on their cumulative correlation coefficients.
        """
        data = self.MinmaxVARIABLES(number)
        abs_data = data.corr().abs()
        cumprod_data = abs_data.cumprod() ** (1 / len(abs_data))
        cumprod_sort_index = pd.Series(cumprod_data.iloc[:, -1].sort_values(ascending=False).index)
        return data[cumprod_sort_index]
