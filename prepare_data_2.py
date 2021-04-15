import os
from pathlib import Path
import h5py
import numpy as np


class prepare_data_2():
    def __init__(self, data_path):
        self.data_path = data_path
        # self.data_split_11 = None
        self.data_split = None

    def split_data_and_save(self, segment_length=2, overlap=0, sampling_rate=1000, save=True):
        base_path = Path(self.data_path)
        hdf_path_list = base_path.glob("*.hdf")
        # hdf_path_list = base_path.glob("sub2_002_1_data_split.hdf")
        for hdf_path in hdf_path_list:
            self.data_split = None
            self.split_data(hdf_path, segment_length, overlap, sampling_rate)
            if self.data_split is None:
                print("split_data_and_save error. error message: {} is empty!".format(hdf_path))
                continue
            if save:
                save_base_path = Path(os.getcwd())
                file_name = Path(hdf_path).stem
                if not Path.exists(save_base_path / Path("data_2")):
                    Path(save_base_path / Path("data_2")).mkdir(parents=True)
                data_save_file_path = save_base_path / Path("data_2/" + file_name + ".hdf")
                self.save_hdf_data({"data": self.data_split}, data_save_file_path)

    def save_hdf_data(self, data_dict, save_file_path):
        try:
            f = h5py.File(save_file_path, 'w')
            for k, v in data_dict.items():
                f[k] = v
            f.close()
        except Exception as e:
            print("save_hdf_data error. error message: {}".format(e))

    def get_end_index(self, elem):
        return int(elem.split("_")[1])

    def split_data(self, hdf_path, segment_length, overlap, sampling_rate):
        try:
            f = h5py.File(hdf_path, 'r')
            data_step = int(segment_length * sampling_rate * (1 - overlap))
            data_segment = sampling_rate * segment_length
            keys_list = [i for i in f.keys()]
            keys_list.sort(key=self.get_end_index)
            data_split = []
            for key in keys_list:
                data = f[key]
                number_segment = int((data.shape[1] - data_segment) // (data_step)) + 1
                for i in range(number_segment):
                    data_split.append(data[:, i * data_step:i * data_step + data_segment])
            self.data_split = np.stack(data_split, axis=0)
        except Exception as e:
            print("split_data error: {}. error message: {}".format(hdf_path, e))


if __name__ == '__main__':
    pre = prepare_data_2("/Users/weiyong/Desktop/eeg/save_2")
    pre.split_data_and_save()
    # pre.split_data("/Users/weiyong/Desktop/eeg/save_2/sub1_005_2_data_split.hdf",2,0,1000)
