from unittest import TestCase
from Segmenter.Segmenter import Segmenter
from Gen1CutFileHandler import Gen1CutFileHandler
from Visualizer import Visualizer


class TestSegmenter(TestCase):

    def setUp(self) -> None:
        self.filepath_gen1 = r"C:\Data\Gen1CutFile\SilentTools_00410_20211130-143236.cut"
        self.gen1CutFileHandler = Gen1CutFileHandler()
        self.segmenter = Segmenter()
        self.visualizer = Visualizer()

    def test_segment(self) -> None:
        self.gen1CutFileHandler.process_file(self.filepath_gen1, resolution_ms=250)
        df = self.gen1CutFileHandler.synchronized_data
        df_segmented = self.segmenter.segment(df)
        fig = self.visualizer.plot_segmented(df_segmented, df, use_plotly=False)
        fig.show()
        # fig = self.visualizer.lineplot(df, line_color="black", line_width=.5, use_plotly=False)
        # fig.show()
