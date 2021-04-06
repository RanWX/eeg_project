import matplotlib.pyplot as plt
import plot.data_tools as data_tools

data_path = '/Users/weiyong/PycharmProjects/eeg_project_test/result/100epochpatient.txt'
train_acc, train_loss, val_acc, val_loss = data_tools.get_acc("", data_path)
x_data = [i for i in range(len(train_acc))]
fig1 = plt.figure()
ln1, = plt.plot(x_data, train_acc, color='red', linewidth=2.0, linestyle='--',label="train_acc")
ln2, = plt.plot(x_data, val_acc, color='blue', linewidth=3.0, linestyle='-.',label="val_acc")
plt.legend()
plt.title("acc")  # 设置标题及字体

plt.savefig("acc_patient.png")
fig2 = plt.figure()
ln3, = plt.plot(x_data, train_loss, color='red', linewidth=2.0, linestyle='--',label="train_loss")
ln4, = plt.plot(x_data, val_loss, color='blue', linewidth=3.0, linestyle='-.',label="val_loss")
plt.legend()
plt.title("loss")  # 设置标题及字体
plt.savefig("loss_patient.png")



