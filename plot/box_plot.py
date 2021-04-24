import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plot.data_tools as data_tools

# 设置中文和负号正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 设置图形的显示风格
# plt.style.use('ggplot')
# 绘图：daily_Ionset_r_c1_predicted的箱线图
# to exp 1
# data = data_tools.get_batch_N_fold_acc("../result_total_10_fold/", -1)
# to exp 2
data = data_tools.get_batch_N_fold_acc("../result_healthy_10_fold/", 1)
df = pd.DataFrame(data)
df.plot.box(title="accuracy in different type",
            showmeans=True,
            patch_artist=True,
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
            meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
            medianprops={'linestyle': '--', 'color': 'orange'}
            )
# plt.grid(linestyle="--", alpha=0.3)
plt.xlabel("type")
plt.ylabel("accuracy(%)")
plt.savefig('./results_imgs.png', bbox_inches='tight')

