from unittest import TestCase
from Gen1CutFileHandler import Gen1CutFileHandler
from CutFileHandler import CutFileHandler
from Segmenter.BreakPointDetector import BreakPointDetector
from Visualizer import Visualizer


class TestSegmenter(TestCase):

        def setUp(self) -> None:
            self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
            self.gen1CutFileHandler = Gen1CutFileHandler()
            self.visualizer = Visualizer()
            self.breakpointDetector = BreakPointDetector()

        def test_segment_data(self) -> None:
            self.gen1CutFileHandler.process_file(self.filePath, resolution_ms=250)
            df = self.gen1CutFileHandler.df_sync
            signal = df.iloc[:, 1:].to_numpy()
            result = self.breakpointDetector.fit(signal, pen=100000, model_type="Pelt", model="l1", min_size=1, jump=1)
            fig = self.visualizer.segmentplot(df, result, line_color="black", use_plotly=False)
            fig.show()
            # fig.write_html("temp_figure.html", auto_open=True)
            print("he")

        # def test_segment_cut_file(self) -> None:
        #     self.filepath = r"C:\Users\nq9093\Downloads\JorgensData\Heat Treated HRC46_SS2541_TR-DC1304-F 4415.cut"
        #     self.cutfile_handler = CutFileHandler()
        #     self.cutfile_handler.process_file(self.filepath, resolution_ms=250)
        #     signal = self.cutfile_handler.df_sync.iloc[:, 1:].to_numpy()
        #     result = self.breakpointDetector.fit(signal, pen=100000, model_type="Pelt", detect_transitions=True, window_size=5, model="l1", min_size=1, jump=1)
        #     fig = self.visualizer.segmentplot(self.cutfile_handler.df_sync, result, line_color="black", use_plotly=False)
        #     fig.show()
        #     pass