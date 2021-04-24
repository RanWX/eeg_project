from pathlib import Path
import os, re


def get_acc(data_file, save=""):
    # data_path = Path(os.getcwd()).parent / Path("data")
    with open(data_file) as f:
        lines = f.readlines()
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []
        for line in lines:
            if "Epoch" in line:
                search_content = re.search("^Epoch.+: (.+),.+: (.+)", line)
                train_acc.append(search_content.group(2))
                train_loss.append(search_content.group(1))
            elif "Evaluation" in line:
                search_content = re.search("Evaluation.+:(.+), .+: (.+)", line)
                val_acc.append(search_content.group(2))
                val_loss.append(search_content.group(1))
    return train_acc, train_loss, val_acc, val_loss


def get_best_acc(data_file, save=""):
    test_loss = None
    test_acc = None
    with open(data_file) as f:
        content = f.read()
        search_content = re.search("Test Loss:([0-9]{1,}[.][0-9]*).+([0-9]{1,}[.][0-9]*)", content)
        test_loss = search_content.group(1)
        test_acc = search_content.group(2)
    return test_acc, test_loss


def get_N_fold_acc(data_file,filename_index, save=""):
    '''
    获取文件的正确率
    :param data_file:
    :param save:
    :return: {name1: [acc1, acc2...], name2:[...]}
    '''
    acc_list = None
    with open(data_file) as f:
        content = f.read()
        pattern = re.compile("Test Loss:.+Acc: ([0-9]{1,}[.][0-9]*)")
        acc_list = pattern.findall(content)
    acc_list = [float(i)*100 for i in acc_list]
    file_name = Path(data_file).stem.split("_")[filename_index]
    return file_name, acc_list


def get_batch_N_fold_acc(data_dir, filename_index,save=""):
    file_list = Path(data_dir).glob("*.txt")
    data_dict = {}
    for file in file_list:
        file_name, acc_list = get_N_fold_acc(str(file), filename_index)
        data_dict[file_name] = acc_list
    return data_dict


if __name__ == '__main__':
    data_path = '/Users/weiyong/PycharmProjects/eeg_project_test/result_healthy_10_fold'
    # get_N_fold_acc(data_path)
    get_batch_N_fold_acc(data_path,1)
