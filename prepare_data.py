import os
from pathlib import Path
import h5py
import numpy as np


class prepare_data():
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_split_11 = None
        self.data_split_22 = None

    def split_data_and_save(self, segment_length=2, overlap=0, sampling_rate=1000, save=True):
        base_path = Path(self.data_path)
        hdf_path_list = base_path.glob("*.hdf")
        # hdf_path_list = base_path.glob("sub2_002_1_data_split.hdf")
        for hdf_path in hdf_path_list:
            self.data_split_11 = None
            self.data_split_22 = None
            self.split_data(hdf_path, segment_length, overlap, sampling_rate)
            if self.data_split_11 is None or self.data_split_22 is None:
                print("split_data_and_save error. error message: {} is empty!".format(hdf_path))
                continue
            if save:
                save_base_path = Path(os.getcwd())
                file_name = Path(hdf_path).stem
                if not Path.exists(save_base_path / Path("data/prepare_data")):
                    Path(save_base_path / Path("data/prepare_data")).mkdir(parents=True)
                data_11_save_file_path = save_base_path / Path("data/prepare_data/" + file_name + "_11.hdf")
                data_22_save_file_path = save_base_path / Path("data/prepare_data/" + file_name + "_22.hdf")
                self.save_hdf_data({"data": self.data_split_11}, data_11_save_file_path)
                self.save_hdf_data({"data": self.data_split_22}, data_22_save_file_path)

    def save_hdf_data(self, data_dict, save_file_path):
        try:
            f = h5py.File(save_file_path, 'w')
            for k, v in data_dict.items():
                f[k] = v
            f.close()
        except Exception as e:
            print("save_hdf_data error. error message: {}".format(e))

    def split_data(self, hdf_path, segment_length, overlap, sampling_rate):
        try:
            f = h5py.File(hdf_path, 'r')
            data_step = int(segment_length * sampling_rate * (1 - overlap))
            data_segment = sampling_rate * segment_length
            data_split_11 = []
            data_split_22 = []
            for key in f.keys():
                data = f[key]
                number_segment = int((data.shape[0] - data_segment) // (data_step)) + 1
                for i in range(number_segment):
                    if "11" in key:
                        data_split_11.append(data[i * data_step:i * data_step + data_segment, :])
                    elif "22" in key:
                        data_split_22.append(data[i * data_step:i * data_step + data_segment, :])
            self.data_split_11 = np.stack(data_split_11, axis=0)
            self.data_split_22 = np.stack(data_split_22, axis=0)
        except Exception as e:
            print("split_data error: {}. error message: {}".format(hdf_path, e))


if __name__ == '__main__':
    pre = prepare_data("/Users/weiyong/Desktop/eeg/save")
    pre.split_data_and_save()
    # pre.split_data("/Users/ranshuang/Desktop/eeg/save/sub1_068_2_data_split.hdf")
