"""
This class creates an abstract representation for the data model.

Author: Yaolin Ge
Date: 2024-10-18
"""
import pandas as pd


class DataModel:

    def __init__(self) -> None:
        self.columns = ['x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1']
        self.dataframe = pd.DataFrame(columns=self.columns)


