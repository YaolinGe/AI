import os
from unittest import TestCase

from Components.SegmenterAnalysisPage import visualizer
from CutFileMerger import CutFileMerger
from Visualizer import Visualizer


class TestCutFileMerger(TestCase):

    def setUp(self) -> None:
        self.cut_file_merger = CutFileMerger()
        self.visualizer = Visualizer()

    def test_merge_cut_files(self) -> None:
        folder_path = r"C:\Data\MissyDataSet\Missy_Disc1\Cutfiles"
        filenames = os.listdir(folder_path)
        filenames = [os.path.join(folder_path, filename) for filename in filenames if filename.endswith('.cut')]
        self.cut_file_merger.merge_cut_files(filenames, resolution_ms=250)
        df = self.cut_file_merger.df_merged
        fig = visualizer.lineplot(df, line_color="black", line_width=.5, use_plotly=False)
        fig.show()
        fig.show()