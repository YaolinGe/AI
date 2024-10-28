from unittest import TestCase
from Gen1CSVHandler import Gen1CSVHandler
from CutFileHandler import CutFileHandler
from Segmenter import Segmenter
from Visualizer import Visualizer


class TestSegmenter(TestCase):

        def setUp(self) -> None:
            self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
            self.gen1CSVHandler = Gen1CSVHandler()
            self.visualizer = Visualizer()

        # def test_segment_data(self) -> None:
        #     self.gen1CSVHandler.process_file(self.filePath, resolution_ms=100)
        #     df = self.gen1CSVHandler.df_sync
        #     # fig = self.visualizer.lineplot(df, line_color="white", line_width=.5, use_plotly=False, text_color="white")
        #     # fig.show()
        #     signal = df.iloc[:, 1:].to_numpy()
        #     # self.segmenter = Segmenter(model_type="BottomUp", model="l1")
        #     # self.segmenter = Segmenter(model_type="Pelt", model="l2")
        #     self.segmenter = Segmenter(model_type="BottomUp", model="l1", min_size=10, jump=1)
        #     result = self.segmenter.fit(signal, pen=5000)
        #     # fig = self.segmenter.plot_results()
        #     fig = self.visualizer.segmentplot(df, result, line_color="black", use_plotly=False)
        #     fig.show()
        #     # fig.write_html("temp_figure.html", auto_open=True)
        #     print("he")

        def test_segment_cut_file(self) -> None:
            self.filepath = r"C:\Users\nq9093\Downloads\JorgensData\Heat Treated HRC46_SS2541_TR-DC1304-F 4415.cut"
            self.cutfile_handler = CutFileHandler()
            self.cutfile_handler.process_file(self.filepath, resolution_ms=100)
            self.segmenter = Segmenter(model_type="BottomUp", model="l1", min_size=10, jump=1)
            signal = self.cutfile_handler.df_sync.iloc[:, 1:].to_numpy()
            result = self.segmenter.fit(signal, pen=5000)
            fig = self.visualizer.segmentplot(self.cutfile_handler.df_sync, result, line_color="black", use_plotly=False)
            fig.show()
            pass