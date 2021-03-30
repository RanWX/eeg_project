import re, os
from pathlib import Path
import numpy as np
import h5py


def split_dataset_by_proportion(event, src_path="", seed=0, **proportion):
    '''
    根据需要将数据按比例分割
    :param event: "11", "22" or "*"(11&22)
    :param src_path: data path
    :param type: train
    :param proportion: the proportion of train, test and val
    :param seed:
    :return: train_data_list: [{"data_path":"", "class_name":"", "class_index":0, "data_matrix":""}], val_data_list: [{"data_path":"", "class_name":"", "class_index":0, "data_matrix":""}]
    '''
    try:
        base_path = Path(src_path) if src_path != "" else (Path(os.getcwd()) / "data")
        hdf_path_list = base_path.glob("*_{}.hdf".format(str(event)))
        data_healthy_list = []
        data_unhealthy_list = []
        # count = 0
        for i in hdf_path_list:
            # if count > 100: break
            # count = count + 1
            file_name = i.stem
            print(file_name)
            class_name, class_index = _get_class_info(file_name)
            data_path = base_path / Path(i)
            split_data = h5py.File(data_path, 'r')["data"]
            for j in range(split_data.shape[0]):
                data_dict = {"data_path": data_path, "class_name": class_name, "class_index": class_index,
                             "data_matrix": split_data[j, :, :]}
                if class_index == 0:
                    data_unhealthy_list.append(data_dict)
                elif class_index == 1:
                    data_healthy_list.append(data_dict)
        np.random.seed(seed)
        healthy_index_perm_list = np.random.permutation(len(data_healthy_list))
        unhealthy_index_perm_list = np.random.permutation(len(data_unhealthy_list))

        healthy_train_len = int(len(data_healthy_list) * proportion["train"])
        healthy_val_len = int(len(data_healthy_list) * proportion["val"])
        healthy_test_len = int(len(data_healthy_list) * proportion["test"])

        unhealthy_train_len = int(len(data_unhealthy_list) * proportion["train"])
        unhealthy_val_len = int(len(data_unhealthy_list) * proportion["val"])
        unhealthy_test_len = int(len(data_unhealthy_list) * proportion["test"])

        train_data_list = [data_healthy_list[i] for i in healthy_index_perm_list[:healthy_train_len]] + [
            data_unhealthy_list[j] for j in unhealthy_index_perm_list[:unhealthy_train_len]]
        test_data_list = [data_healthy_list[i] for i in
                          healthy_index_perm_list[healthy_train_len:healthy_train_len + healthy_test_len]] + [
                             data_unhealthy_list[j] for j in
                             unhealthy_index_perm_list[unhealthy_train_len:unhealthy_train_len + unhealthy_test_len]]
        val_data_list = [data_healthy_list[i] for i in
                         healthy_index_perm_list[
                         healthy_train_len + healthy_test_len:healthy_train_len + healthy_test_len + healthy_val_len]] + [
                            data_unhealthy_list[j] for j in unhealthy_index_perm_list[
                                                            unhealthy_train_len + unhealthy_test_len:unhealthy_train_len + unhealthy_test_len + unhealthy_val_len]]
        return train_data_list, val_data_list, test_data_list

    except Exception as e:
        print("split_dataset_by_proportion error. error message is: {}".format(e))


def get_data_and_labels_with_batchsize(data_list, batch_size=5, seed=0):
    '''
    获得一个echo的所有batch
    :param data_list:
    :param batch_size: batch大小
    :param seed:
    :return: "batch_data":[[], []...], "batch_label":[[],[]...]
    '''
    np.random.seed(seed)
    data_index_perm_list = np.random.permutation(len(data_list))
    batch_data_list = []
    batch_label_list = []
    num_batch = int(np.ceil(len(data_index_perm_list) / batch_size))
    for batch_id in range(num_batch):
        offset = batch_id * batch_size
        this_index_list = data_index_perm_list[offset:(offset + batch_size)]
        batch_data = []
        batch_label = []
        for i in this_index_list:
            batch_data.append(data_list[i]["data_matrix"])
            batch_label.append(data_list[i]["class_index"])
        batch_data_list.append(batch_data)
        batch_label_list.append(batch_label)
    assert len(batch_data_list) == len(batch_label_list) == num_batch
    return batch_data_list, batch_label_list


def _get_class_info(str):
    class_type = re.search("^sub\d", str).group(0)
    class_name = None
    class_index = None
    if class_type == "sub1":
        class_name = "unhealthy"
        class_index = 0
    elif class_type == "sub2":
        class_name = "healthy"
        class_index = 1
    else:
        raise RuntimeError("class type error!")
    return class_name, class_index


if __name__ == '__main__':
    proportion = {"train": 0.6, "test": 0.2, "val": 0.2}
    train_data_list, val_data_list, test_data_list = split_dataset_by_proportion("11", src_path="", seed=0,
                                                                                 **proportion)
    a, b = get_data_and_labels_with_batchsize(train_data_list, batch_size=5, seed=0)
    c = 0
