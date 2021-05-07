import numpy as np
import matplotlib.pyplot as plt

import plot.data_tools as data_tools

data = data_tools.get_batch_N_fold_acc_key_to_int("../result_1122_10_fold_1/", -1)
x,y = [],[]
for i in sorted(data) :
    print((i, data[i]), end ="\n")
    x.append(i)
    y.append(np.mean(data[i]))

z1 = np.polyfit(x, y, 3)  # 用4次多项式拟合
p1 = np.poly1d(z1)
print(p1)  # 在屏幕上打印拟合多项式
yvals = p1(x)  # yvals=np.polyval(z1,x)
plot1 = plt.plot(x, y, '*', label='original values')
plot2 = plt.plot(x, yvals, 'r', label='fitting values')
plt.xlabel('timepoint(s)')
plt.ylabel('accuracy(%)')

plt.legend(loc=4)
# plt.title('')
plt.savefig('./results_imgs_3.png', bbox_inches='tight')
# plt.show()
