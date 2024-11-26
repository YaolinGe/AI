import os
from unittest import TestCase
from DataAnnotator import DataAnnotator


class TestDataAnnotator(TestCase):

    def setUp(self) -> None:
        self.data_annotator = DataAnnotator(cache_folder=".cache", autosave_interval=5.0)

    def test_data_annotations(self) -> None:
        filepath = r"annotations\test.csv"
        self.data_annotator.add_annotation(filepath, 0, 1000, "test")
        self.data_annotator.add_annotation(filepath, 1000, 2000, "incut")
        self.data_annotator.add_annotation(filepath, 2000, 3000, "outofcut")
        self.data_annotator.add_annotation(filepath, 3000, 4000, "process")
        self.data_annotator.add_annotation(filepath, 4000, 5000, "test")
        self.data_annotator.add_annotation(filepath, 1000, 1200, "incut")