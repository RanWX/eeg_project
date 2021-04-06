from pathlib import Path
import os, re


def get_acc(save_path, data_file):
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


if __name__ == '__main__':
    data_path = '/Users/weiyong/PycharmProjects/eeg_project_test/result/100epoch22.txt'
    get_acc("", data_path)
