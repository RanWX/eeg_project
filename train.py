import torch


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


if __name__ == '__main__':
    train = train()
