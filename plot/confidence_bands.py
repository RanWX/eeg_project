import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plot.data_tools as data_tools

data = data_tools.get_batch_N_fold_acc_key_to_int("../result_1122_10_fold_1_20210512/", -1)
x, y = [], []
x1, y1 = [], []
for i in sorted(data):
    # print((i, data[i]), end="\n")
    x1.append(i)
    y1.append(np.mean(data[i]))
    # if np.mean(data[i]) < 92 and i > 300:
    #     continue
    y.append(np.mean(data[i]))
    x.append(i)

# fit a linear curve an estimate its y-values and their error.
x = np.array(x)
y = np.array(y)
a, b, c, d, e = np.polyfit(x, y, 4)
y_est = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
# print(y)
# y_err = x.std()
# print(y_err)
y_err = y.std() * np.sqrt(1 / len(y) +
                          (y - y.mean()) ** 2 / np.sum((y - y.mean()) ** 2))

fig, ax = plt.subplots()
ax.plot(x, y_est, '-', label='fitting values')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.1)
ax.plot(x, y, '*', label='original values')
plt.legend(loc=4)
plt.savefig('./results_imgs_xx.png', bbox_inches='tight')
