"""
Annotator module

Created on 2024-11-26
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import os
import pandas as pd
from Visualizer import Visualizer


class Annotator:

    def __init__(self) -> None:
        self.annotations = pd.read_csv(os.path.join("annotations", "df_disk1_annotation.csv"))
        self.visualizer = Visualizer()
        self.df = pd.read_csv(os.path.join("datasets", "df_disk1.csv"))

    def visualize_annotations(self):

