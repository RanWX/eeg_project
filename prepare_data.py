import os
from pathlib import Path
import h5py
import numpy as np


# cut and save

class prepare_data():
    def __init__(self, mode, event_type, data_path):
        self.mode = mode
        self.data_path = data_path
        self.event_type = event_type

    def split_data_and_save(self, segment_length=1, overlap=0, sampling_rate=1000, save=True):
        base_path = Path(self.data_path) if self.event_type == "all" else Path(
            Path(self.data_path) / Path(self.event_type))
        npy_path_list = base_path.glob("**/*/*.npy")
        data_split = []
        label_split = []
        for i in npy_path_list:
            print(i)

        # save_path = Path(os.getcwd())
        # if not Path.exists(save_path / Path("data/prepare_data")):
        #     Path(save_path / Path("data/prepare_data")).mkdir(parents=True)

        # 取病人发病期数据和正常人对应的时间的数据

    def split_data(self, npy_path):
        pass


if __name__ == '__main__':
    pre = prepare_data(1, "all", "/Users/ranshuang/Desktop/eeg/save")
    pre.split_data_and_save()
