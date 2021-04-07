import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
This is the models of TSception and its variant

To use the models, please manage the data into
the dimension of(mini-batch, 1, EEG-channels,data point)
before feed the data into forward()

For more details about the models, please refer to our paper:

Yi Ding, Neethu Robinson, Qiuhao Zeng, Dou Chen, Aung Aung Phyo Wai, Tih-Shih Lee, Cuntai Guan,
"TSception: A Deep Learning Framework for Emotion Detection Useing EEG"(IJCNN 2020)

'''


################################################## TSception ######################################################
class TSception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hiden, dropout_rate):
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[1] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0]), 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0] * 0.5), 1), stride=(int(input_size[0] * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_size(self, input_size):
        # here we use and array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, 1, input_size[0], input_size[1]))

        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()


######################################### Temporal ########################################
class Tception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, hiden, dropout_rate):
        # input_size: channel x datapoint
        super(Tception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[1] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))

        self.BN_t = nn.BatchNorm2d(num_T)

        size = self.get_size(input_size, sampling_rate, num_T)
        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_size(self, input_size, sampling_rate, num_T):
        data = torch.ones((1, 1, input_size[0], input_size[1]))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = out.view(out.size()[0], -1)
        return out.size()


############################################ Spacial ########################################
class Sception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_S, hiden, dropout_rate):
        # input_size: channel x datapoint
        super(Sception, self).__init__()

        self.Sception1 = nn.Sequential(
            nn.Conv2d(1, num_S, kernel_size=(int(input_size[0]), 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(1, num_S, kernel_size=(int(input_size[0] * 0.5), 1), stride=(int(input_size[0] * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))

        self.BN_s = nn.BatchNorm2d(num_S)

        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())

    def forward(self, x):
        y = self.Sception1(x)
        out = y
        y = self.Sception2(x)
        out = torch.cat((out, y), dim=2)
        out = self.BN_s(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_size(self, input_size):
        data = torch.ones((1, 1, input_size[0], input_size[1]))
        y = self.Sception1(data)
        out = y
        y = self.Sception2(data)
        out = torch.cat((out, y), dim=2)
        out = self.BN_s(out)
        out = out.view(out.size()[0], -1)
        return out.size()


############################################ EEGNET ########################################
# 64 * 2000
class EEGNet(nn.Module):
    def __init__(self, class_num, chan, dropout1, dropout2, time_point, samples):
        super(EEGNet, self).__init__()
        self.first_layer = nn.Sequential(
            # TODO kernel_size
            nn.ZeroPad2d((499, 0, 0, 0)),
            nn.Conv2d(1, 16, kernel_size=(1, int(samples / 2)), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16)  # 16 C T
        )
        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(16, 2 * 16, kernel_size=(chan, 1), stride=(1, 1), groups=16, bias=False),  # 32 1 T
            nn.BatchNorm2d(32),  # 32 1 T
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),  # 32 1 T//4
            nn.Dropout(p=dropout1)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout2)
        )
        self.out = nn.Linear((32 * (time_point // 32)), class_num)

    def forward(self, x):
        x = self.first_layer(x)
        # print("first_layer: {}".format(x.shape))
        x = self.depth_wise_conv(x)
        # print("depth_wise_conv: {}".format(x.shape))
        x = self.separable_conv(x)
        # print("separable_conv: {}".format(x.shape))
        x = x.view(x.size(0), -1)
        # print("separable_conv next: {}".format(x.shape))
        x = self.out(x)
        # print("out: {}".format((x.shape)))
        x = F.softmax(x, dim=1)
        # print("out lastu: {}".format((x.shape)))
        return x


if __name__ == "__main__":
    # model = TSception(2, (4, 1024), 256, 9, 6, 128, 0.2)
    # model = Sception(2,(4,1024),256,6,128,0.2)
    # model = Tception(2,(4,1024),256,9,128,0.2)
    # print(model)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)
    # input = torch.randn(32, 1, 22, 1125)

    eeg_net = EEGNet(2, 62, 0.25, 0.25, 2000, 1000)
    input1 = torch.randn(1, 1, 62, 2000)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    eeg_net.train()
    output = eeg_net(input1)
