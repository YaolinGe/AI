from unittest import TestCase
from BatchAnalyzer import BatchAnalyzer
from Gen1CutFileHandler import Gen1CutFileHandler
from Visualizer import Visualizer
import os


class TestStatisticalReferenceBuilder(TestCase):

    def setUp(self) -> None:
        self.visualizer = Visualizer()
        self.batch_analyzer = BatchAnalyzer()

    # def test_gen1_cutfile_batch_analysis(self):
    #     self.filepath_gen1 = r"C:\Users\nq9093\OneDrive - Sandvik\Data\Gen1CutFile\SilentTools_00410_20211130-143236.cut"
    #     self.folder_path = os.path.dirname(self.filepath_gen1)
    #     files = os.listdir(self.folder_path)
    #     self.filenames = []
    #     for filename in files:
    #         if filename.endswith(".cut"):
    #             self.filenames.append(os.path.join(self.folder_path, filename))
    #
    #     result = self.batch_analyzer.analyze_batch_cutfiles(self.filenames, resolution_ms=250, is_gen1=True)
    #     self.visualizer.plot_batch_confidence_interval(result['segment_1'], line_color="black", line_width=.5, use_plotly=True, sync=True).show()
    #     print("hello")

    def test_cutfile_batch_analysis(self):
        self.folder_path_gen2 = r"C:\Users\nq9093\OneDrive - Sandvik\Data\JorgensData\batch"
        files = os.listdir(self.folder_path_gen2)
        self.filenames = []
        for filename in files:
            if filename.endswith(".cut"):
                self.filenames.append(os.path.join(self.folder_path_gen2, filename))
        result = self.batch_analyzer.analyze_batch_cutfiles(self.filenames, resolution_ms=250, is_gen1=False)
        self.visualizer.plot_batch_confidence_interval(result['segment_1'], line_color="black", line_width=.5, use_plotly=False, sync=True).show()
        print("hello")
