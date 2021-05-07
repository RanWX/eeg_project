import numpy as np

import plot.data_tools as data_tools

data = data_tools.get_batch_N_fold_acc("../result_total_10_fold_bak/", -1)
for k, v in data.items():
    mean = np.mean(v)
    std = np.std(v)
    print("{}: , mean: {}".format(k, str(mean)))
    print("{}: , std: {}".format(k, str(std)))
