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


if __name__ == '__main__':
    data_path = '/Users/weiyong/PycharmProjects/eeg_project_test/result/100epoch22.txt'
    get_best_acc(data_path)
