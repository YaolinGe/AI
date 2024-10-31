from unittest import TestCase
from BatchAnalyzer import BatchAnalyzer
from Gen1CutFileHandler import Gen1CutFileHandler
from Visualizer import Visualizer
import os


class TestStatisticalReferenceBuilder(TestCase):

    def setUp(self) -> None:
        self.filepath_gen1 = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
        self.gen1CutFileHandler = Gen1CutFileHandler()
        # self.filepath_gen2 = r"C:\Users\nq9093\Downloads\JorgensData\Heat Treated HRC48_SS2541_TR-DC1304-F 4415.cut"
        # self.cutfile_handler = CutFileHandler()
        self.visualizer = Visualizer()
        self.batch_analyzer = BatchAnalyzer()

        self.folder_path = os.path.dirname(self.filepath_gen1)
        files = os.listdir(self.folder_path)
        self.filenames = []
        for filename in files:
            if filename.endswith(".cut"):
                self.filenames.append(os.path.join(self.folder_path, filename))

    def test_statistical_calculation(self):
        # self.gen1CSVHandler.process_file(self.filepath_gen1, resolution_ms=250)
        # df = self.gen1CSVHandler.df_sync
        # signal = df.iloc[:, 1:].to_numpy()
        # result = self.segmenter.fit(signal, pen=100000, model_type="Pelt", model="l1", min_size=1, jump=1)
        # fig = self.visualizer.segmentplot(df, result, line_color="black", use_plotly=False)
        # fig.show()
        # self.statisticalReferenceBuilder.segment_all_data_frames(self.filenames)
        # result = self.statisticalReferenceBuilder.calculate_confidence_interval()
        result = self.batch_analyzer.analyze_batch_cutfiles(self.filenames)
        self.visualizer.plot_statistical_reference(result['segment_1'], line_color="black", line_width=.5, use_plotly=True, sync=True).show()
        # self.visualizer.lineplot(result['segment_0']['average'], line_color="black", line_width=.5, use_plotly=False).show()
        # self.visualizer.lineplot(result['segment_0']['std'], line_color="black", line_width=.5, use_plotly=False).show()
        print("hello")
