from pathlib import Path
import os
import plot.data_tools as data_tools
import matplotlib.pyplot as plt

# result_path = Path(os.getcwd()).parent / Path("result_span_11")
result_path = Path(os.getcwd()).parent / Path("result_1122_same")
data_path = Path(result_path)
result_list = data_path.glob("*.txt")

y_data_acc = []
x = []
for item in result_list:
    x_data = int(item.stem.split("_")[3].rstrip('s'))
    if (x_data / 10) % 2 == 1:
        continue
    test_acc, test_loss = data_tools.get_best_acc(str(item))
    y_data_acc.append(test_acc)
    # x_data = int(item.stem.split("_")[2].rstrip('s'))
    x.append(x_data)

plt.figure(dpi=128, figsize=(12, 8))
plt.title('test acc in different time span (event 11&22)')
plt.xlabel('time span(s)')
plt.ylabel('acc')
plt.ylim(0.55, 1)
plt.bar(x, y_data_acc, width=5)
for a, b in zip(x, y_data_acc):
    plt.text(a, float(b) + 0.008, b, ha='center', va='bottom', fontsize=7)
plt.savefig("bar_acc_1.png")
