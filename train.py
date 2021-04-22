import datetime, os, time
from pathlib import Path
import torch, h5py
import numpy as np
import data_process
from models import *


class train():
    def __init__(self):
        self.data = None
        self.label = None
        self.result = None
        self.input_shape = None  # should be (eeg_channel, time data point)
        self.model = 'TSception'
        self.cross_validation = 'Session'  # Subject
        self.sampling_rate = 1000

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Parameters: Training process
        self.random_seed = 42
        self.learning_rate = 1e-3
        self.num_epochs = 200
        self.num_class = 2
        self.batch_size = 128
        # TODO
        self.patient = 4

        # Parameters: Model
        self.dropout = 0.3
        self.hiden_node = 128
        self.T = 9
        self.S = 6
        self.Lambda = 1e-6

    def set_parameter(self, cv, model, number_class, sampling_rate,
                      random_seed, learning_rate, epoch, batch_size,
                      dropout, hiden_node, patient,
                      num_T, num_S, Lambda):
        '''
        This is the function to set the parameters of training process and model
        All the settings will be saved into a NAME.txt file
        Input : cv --
                   The cross-validation type
                   Type = string
                   Default : Leave_one_session_out
                   Note : for different cross validation type, please add the
                          corresponding cross validation function. (e.g. self.Leave_one_session_out())

                model --
                   The model you want choose
                   Type = string
                   Default : TSception

                number_class --
                   The number of classes
                   Type = int
                   Default : 2

                sampling_rate --
                   The sampling rate of the EEG data
                   Type = int
                   Default : 256

                random_seed --
                   The random seed
                   Type : int
                   Default : 42

                learning_rate --
                   Learning rate
                   Type : flaot
                   Default : 0.001

                epoch --
                   Type : int
                   Default : 200

                batch_size --
                   The size of mini-batch
                   Type : int
                   Default : 128

                dropout --
                   dropout rate of the fully connected layers
                   Type : float
                   Default : 0.3

                hiden_node --
                   The number of hiden node in the fully connected layer
                   Type : int
                   Default : 128

                patient --
                   How many epoches the training process should wait for
                   It is used for the early-stopping
                   Type : int
                   Default : 4

                num_T --
                   The number of T kernels
                   Type : int
                   Default : 9

                num_S --
                   The number of S kernels
                   Type : int
                   Default : 6

                Lambda --
                   The L1 regulation coefficient in loss function
                   Type : float
                   Default : 1e-6

        '''
        self.model = model
        self.sampling_rate = sampling_rate
        # Parameters: Training process
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = epoch
        self.num_class = number_class
        self.batch_size = batch_size
        self.patient = patient
        self.Lambda = Lambda

        # Parameters: Model
        self.dropout = dropout
        self.hiden_node = hiden_node
        self.T = num_T
        self.S = num_S

        # Save to log file for checking
        if cv == "Leave_one_subject_out":
            file = open("result_subject.txt", 'a')
        elif cv == "Leave_one_session_out":
            file = open("result_session.txt", 'a')
        elif cv == "K_fold":
            file = open("result_k_fold.txt", 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(self.model) +
                   "\n1)number_class:" + str(self.num_class) + "\n2)random_seed:" + str(self.random_seed) +
                   "\n3)learning_rate:" + str(self.learning_rate) + "\n4)num_epochs:" + str(self.num_epochs) +
                   "\n5)batch_size:" + str(self.batch_size) +
                   "\n6)dropout:" + str(self.dropout) + "\n7)sampling_rate:" + str(self.sampling_rate) +
                   "\n8)hiden_node:" + str(self.hiden_node) + "\n9)input_shape:" + str(self.input_shape) +
                   "\n10)patient:" + str(self.patient) + "\n11)T:" + str(self.T) +
                   "\n12)S:" + str(self.S) + "\n13)Lambda:" + str(self.Lambda) + '\n')

        file.close()

    def train_model(self, train_data_list, val_data_list, test_data_list, cv_type, i):
        # TODO: no cuda
        # print('Avaliable device:' + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        torch.manual_seed(self.random_seed)
        # torch.backends.cudnn.deterministic = True
        # Train and validation loss
        losses = []
        accs = []

        Acc_val = []
        Loss_val = []

        Acc_test = []

        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs
        # input_size: EEG channel x datapoint
        self.input_shape = train_data_list[0]["data_matrix"].shape
        # build the model
        if self.model == 'Sception':
            model = Sception(num_classes=self.num_class, input_size=self.input_shape,
                             sampling_rate=self.sampling_rate, num_S=self.S,
                             hiden=self.hiden_node, dropout_rate=self.dropout)
        elif self.model == 'Tception':
            model = Tception(num_classes=self.num_class, input_size=self.input_shape,
                             sampling_rate=self.sampling_rate, num_T=self.T,
                             hiden=self.hiden_node, dropout_rate=self.dropout)
        elif self.model == 'TSception':
            model = TSception(num_classes=self.num_class, input_size=self.input_shape,
                              sampling_rate=self.sampling_rate, num_T=self.T, num_S=self.S,
                              hiden=self.hiden_node, dropout_rate=self.dropout)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss_fn = nn.CrossEntropyLoss()

        model = model.to(self.device)
        loss_fn = loss_fn.to(self.device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        train_step = self.make_train_step(model, loss_fn, optimizer)

        # load the data
        train_batch_data_list, train_batch_label_list = data_process.get_data_and_labels_with_batchsize(train_data_list,
                                                                                                        self.batch_size)
        test_batch_data_list, test_batch_label_list = data_process.get_data_and_labels_with_batchsize(test_data_list,
                                                                                                      self.batch_size)
        val_batch_data_list, val_batch_label_list = data_process.get_data_and_labels_with_batchsize(val_data_list,
                                                                                                    self.batch_size)
        train_list = [train_batch_data_list, train_batch_label_list]
        test_list = [test_batch_data_list, test_batch_label_list]
        val_list = [val_batch_data_list, val_batch_label_list]

        ######## Training process ########
        # print("start train")
        save_list = []
        save_list.append("************ {} fold **********".format(str(i)))
        acc_max = 0
        patient = 0

        start_time = time.time()
        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            for (x_batch, y_batch) in zip(train_list[0], train_list[1]):
                x_batch = np.array(x_batch)
                x_batch = np.expand_dims(x_batch, axis=1)
                x_batch = torch.from_numpy(x_batch)
                y_batch = torch.from_numpy(np.array(y_batch))
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)

            losses.append(sum(loss_epoch) / len(loss_epoch))
            accs.append(sum(acc_epoch) / len(acc_epoch))
            # print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
            #       .format(epoch + 1, num_epochs, losses[-1], accs[-1]))

            ######## Validation process ########
            val_losses = []
            val_acc = []
            with torch.no_grad():
                for x_batch, y_batch in zip(val_list[0], val_list[1]):
                    x_batch = np.array(x_batch)
                    x_batch = np.expand_dims(x_batch, axis=1)
                    x_batch = torch.from_numpy(x_batch)
                    y_batch = torch.from_numpy(np.array(y_batch))
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    model.eval()

                    yhat = model(x_batch)
                    pred = yhat.max(1)[1]
                    correct = (pred == y_batch).sum()
                    acc = correct.item() / len(pred)
                    val_loss = loss_fn(yhat, y_batch.long())
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)

                Acc_val.append(sum(val_acc) / len(val_acc))
                Loss_val.append(sum(val_losses) / len(val_losses))
                # print('Evaluation Loss:{:.4f}, Acc: {:.4f}'
                #       .format(Loss_val[-1], Acc_val[-1]))

            ######## early stop ########
            Acc_es = Acc_val[-1]

            if Acc_es > acc_max:
                acc_max = Acc_es
                patient = 0
                # print('----Model saved!----')
                torch.save(model, 'max_model.pt')
            else:
                patient += 1
            if patient > self.patient:
                # print('----Early stopping----')
                break

        ######## test process ########
        # print("start test")
        test_losses = []
        test_acc = []
        model = torch.load('max_model.pt')
        with torch.no_grad():
            for (x_batch, y_batch) in zip(test_list[0], test_list[1]):
                x_batch = np.array(x_batch)
                x_batch = np.expand_dims(x_batch, axis=1)
                x_batch = torch.from_numpy(x_batch)
                y_batch = torch.from_numpy(np.array(y_batch))
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                model.eval()

                yhat = model(x_batch)
                pred = yhat.max(1)[1]
                correct = (pred == y_batch).sum()
                acc = correct.item() / len(pred)
                test_loss = loss_fn(yhat, y_batch.long())
                test_losses.append(test_loss.item())
                test_acc.append(acc)

            # print('Test Loss:{:.4f}, Acc: {:.4f}'
            #       .format(sum(test_losses) / len(test_losses), sum(test_acc) / len(test_acc)))
            Acc_test = (sum(test_acc) / len(test_acc))
            save_list.append('Test Loss:{:.4f}, Acc: {:.4f}'
                             .format(sum(test_losses) / len(test_losses), sum(test_acc) / len(test_acc)))

        # save the loss(acc) for plotting the loss(acc) curve
        save_path = Path(os.getcwd())
        if not Path.exists(save_path / Path('Result_model/Leave_one_session_out/history')):
            Path(save_path / Path('Result_model/Leave_one_session_out/history')).mkdir(parents=True)
        if cv_type == "leave_one_session_out":
            filename_callback = save_path / Path('Result_model/Leave_one_session_out/history/' + 'train_history.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['val_acc'] = Acc_val
            save_history['loss'] = losses
            save_history['val_loss'] = Loss_val
            save_history.close()
        time_elapsed = time.time() - start_time
        # print('Training complete in {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(acc_max))
        save_list.append('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        save_list.append('Best train Loss: {:4f}, Acc: {:4f}'.format(losses[-1], accs[-1]))
        save_list.append('Evaluation Loss:{:.4f}, Acc: {:.4f}'.format(Loss_val[-1], Acc_val[-1]))
        save_list.append('Best val Acc: {:4f}'.format(acc_max))
        save_str = "\n".join(save_list)
        return save_str

    def make_train_step(self, model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item() / len(pred)
            # L1 regularization
            loss_r = self.regulization(model, self.Lambda)
            # yhat is in one-hot representation;
            loss = loss_fn(yhat, y.long()) + loss_r
            # loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item(), acc

        return train_step

    def regulization(self, model, Lambda):
        w = torch.cat([x.view(-1) for x in model.parameters()])
        err = Lambda * torch.sum(torch.abs(w))
        return err


if __name__ == '__main__':
    proportion = {"train": 0.6, "test": 0.2, "val": 0.2}
    train = train()
    # set parameters
    train.set_parameter(cv='Leave_one_session_out',
                        model='TSception',
                        number_class=2,
                        sampling_rate=256,
                        random_seed=42,
                        learning_rate=0.001,
                        epoch=50,
                        batch_size=32,
                        dropout=0.3,
                        hiden_node=128,
                        patient=4,
                        num_T=9,
                        num_S=6,
                        Lambda=0.000001)
    k = 10
    type = "11"
    result_list = []
    for i in range(k):
        print("************ {} fold **********".format(str(i)))
        train_data_list, val_data_list, test_data_list = data_process.split_dataset_by_proportion(type, src_path="",
                                                                                                  seed=0,
                                                                                                  **proportion)
        save_str = train.train_model(train_data_list, val_data_list, test_data_list, "leave_one_session_out", i)
        result_list.append(save_str)
    if not Path.exists(Path("result_total_10_fold")):
        Path("result_total_10_fold").mkdir(parents=True)
    with open("result_total_10_fold/result_total_10_fold_{}.txt".format(type, "w"))as f:
        f.write("\n".join(result_list))
