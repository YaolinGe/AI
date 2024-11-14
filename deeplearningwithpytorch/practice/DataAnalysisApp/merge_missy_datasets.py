"""
This script will merge all Missy .cut files into a single .csv file. 
"""

from CutFileMerger import CutFileMerger
import os



if __name__ == "__main__": 
    try:
        missy_dir = r"C:\Data\MissyDataSet\Missy_Disc1\Cutfiles"
        # missy_dir = r"C:\Data\MissyDataSet"
        filenames = os.listdir(missy_dir)
        filenames = [os.path.join(missy_dir, filename) for filename in filenames if filename.endswith('.cut')]

        cut_file_merger = CutFileMerger()
        cut_file_merger.merge_cut_files(filenames, resolution_ms=100, filepath=os.path.join("datasets", "df_missy_disk1.csv"))

    except Exception as e:
        print(f"Error: {e}")
        raise e