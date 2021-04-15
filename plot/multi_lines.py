from pathlib import Path
import os
import plot.data_tools as data_tools
import matplotlib.pyplot as plt

result_path = Path(os.getcwd()).parent / Path("result_span_11")
data_path = Path(result_path)
result_list = data_path.glob("*.txt")
plt.figure()
plt.title('train acc in different time span (event 11)', fontsize=20)
plt.xlabel('time span(s)', fontsize=14)
plt.ylabel('acc', fontsize=14)

for item in result_list:
    train_acc, train_loss, val_acc, val_loss = data_tools.get_acc(str(item))
    plt.plot([i for i in range(len(train_acc))], train_acc, linewidth=1.5)

plt.savefig("train_acc.png")
