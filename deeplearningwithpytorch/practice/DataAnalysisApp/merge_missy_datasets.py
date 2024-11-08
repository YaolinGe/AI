"""
This script will merge all Missy .cut files into a single .csv file. 
"""

from CutFileMerger import CutFileMerger
import os
import logging

# Set up logging
log_dir = ".log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "merge_missy_datasets.log")

print(f"Logging to {log_file}")

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__": 
    try:
        missy_dir = r"C:\Data\MissyDataSet\Missy_Disc2\Cutfiles"
        # missy_dir = r"C:\Data\MissyDataSet"
        filenames = os.listdir(missy_dir)
        filenames = [os.path.join(missy_dir, filename) for filename in filenames if filename.endswith('.cut')]
        
        logging.info(f"Found {len(filenames)} .cut files to merge.")
        
        cut_file_merger = CutFileMerger()
        cut_file_merger.merge_cut_files(filenames, resolution_ms=100, filepath=os.path.join("datasets", "missy_disk2_merged_updated.csv"))
        
        logging.info("Merge completed successfully.")
    except Exception as e:
        logging.error("An error occurred during the merge process.", exc_info=True)