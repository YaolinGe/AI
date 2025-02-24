from unittest import TestCase
import os
from CutFileHandler import CutFileHandler
from Segmenter.BreakPointDetector import BreakPointDetector
from Visualizer import Visualizer


class TestProcessedDataHandler(TestCase):
    def setUp(self):
        self.breakpointDetector = BreakPointDetector()
        self.visualizer = Visualizer()
        self.zone_colors = ['#FFE5E5', '#E5FFE5', '#E5E5FF', '#FFFFE5']  # Light red, green, blue, yellow

    def test_gen1_cutfile_processing(self) -> None:
        filepath = r"C:\Users\nq9093\Downloads\CoroPlus_250217-144651.cut"
        self.cutfile_handler = CutFileHandler(is_gen2=False, debug=True)
        self.cutfile_handler.process_file(filepath, resolution_ms=500)
        df = self.cutfile_handler.df_sync
        df.to_csv("datasets/df_gulbox.csv", index=False)

        # Temporary code to merge the dataset with the annotation
        import pandas as pd
        df = pd.read_csv("datasets/df_gulbox.csv")
        df_annotation = pd.read_csv("annotations/df_gulbox_annotation.csv")

        df['Anomaly'] = False
        for index, row in df_annotation.iterrows():
            df.loc[df['timestamp'].between(row['TStart'], row['TEnd']), 'Anomaly'] = True

        df.to_csv("datasets/df_gulbox_merged.csv", index=False)

        # fig = self.visualizer.lineplot(df, line_color="white", line_width=.5, use_plotly=False)
        # fig.show()

    # def test_gen2_cutfile_processing(self):
    #     # self.filepath = r"C:\Data\MissyDataSet\Missy_Disc2\CutFiles\CoroPlus_241008-133957.cut"
    #     folderpath = r"C:\Data\MissyDataSet\Missy_Disc2\CutFiles"
    #     figpath = os.path.join(os.getcwd(), "fig")
    #     files = os.listdir(folderpath)
    #     files = [os.path.join(folderpath, file) for file in files if file.endswith('.cut')]
    #     # files = files[0]
    #     files = files[:3]
    #     self.cutfile_handler = CutFileHandler(is_gen2=True, debug=False)
    #     for filepath in files:
    #         self.cutfile_handler.process_file(filepath, resolution_ms=500)
    #         # fig = self.visualizer.lineplot(self.cutfile_handler.df_sync, line_color="black", line_width=.5, use_plotly=False)
    #         fig = self.visualizer.lineplot(self.cutfile_handler.df_sync, line_color="black", text_color="black",
    #                                        line_width=.5, use_plotly=True)
    #         fig.write_html(os.path.join(figpath, os.path.basename(filepath) + ".html"))
