import numpy as np
import matplotlib.pyplot as plt

import plot.data_tools as data_tools

data = data_tools.get_batch_N_fold_acc_key_to_int("../result_1122_10_fold_1_20210512/", -1)
x,y= [],[]
x1,y1= [],[]
for i in sorted(data) :
    print((i, data[i]), end ="\n")
    x1.append(i)
    y1.append(np.mean(data[i]))
    if np.mean(data[i]) < 92 and i >300:
        continue
    y.append(np.mean(data[i]))
    x.append(i)


z1 = np.polyfit(x, y, 4)  # 用4次多项式拟合
p1 = np.poly1d(z1)
print(p1)  # 在屏幕上打印拟合多项式
yvals = p1(x)  # yvals=np.polyval(z1,x)
plot1 = plt.plot(x1, y1, '*', label='original values')
plot2 = plt.plot(x, yvals, 'r', label='fitting values')
plt.xlabel('timepoint(s)')
plt.ylabel('accuracy(%)')

plt.legend(loc=4)
# plt.title('')
plt.savefig('./results_imgs_6.png', bbox_inches='tight')
# plt.show()
