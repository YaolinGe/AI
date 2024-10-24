from unittest import TestCase
from Gen1CSVHandler import Gen1CSVHandler


class TestGen1CSVHandler(TestCase):

    def setUp(self) -> None:
        self.gen1CSVHandler = Gen1CSVHandler()

    def test_sync_data(self):
        self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
        self.gen1CSVHandler.process_file(self.filePath)
        # plt.plot(self.gen1CSVHandler.df_accelerometer['timestamp'], self.gen1CSVHandler.df_accelerometer['x2g'])
        # plt.show()
        # plt.plot(self.gen1CSVHandler.df_strain0['timestamp'], self.gen1CSVHandler.df_strain0['value'])
        # plt.show()
        # plt.plot(self.gen1CSVHandler.df_strain1['timestamp'], self.gen1CSVHandler.df_strain1['value'])
        # plt.show()
        self.gen1CSVHandler.print_load_times()
        # self.gen1CSVHandler.sync_data()
        # self.assertIsNotNone(self.gen1CSVHandler.df_sync)
