import errno
import os
import shutil

"""
Class responsible for preparing paths to specified directories and files
"""


class PathHandler:

    @staticmethod
    def create_dir(directory):
        if os.path.exists(directory):
            print("Directory " + directory + " already exists, cleaning...")
            shutil.rmtree(directory)
        os.makedirs(directory)

    @staticmethod
    def prepare_for_file(file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
