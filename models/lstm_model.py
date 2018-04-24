# Imports
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

# Define model
class Model(nn.Module):

    def __init__(self, input_size, lstm_size, lstm_layers, output_size):
        # Call parent
        super(Model, self).__init__()
        # Define parameters
        self.counter = 0
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True) # LSTM_SIZE : Hidden size
        #self.lstm.bias_ih_l0 # Access to tensors

        self.output = nn.Linear(lstm_size, output_size)

    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size)) # lstm_init : tuple, with len 2
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[1].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))
        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        # self.lstm(x, lstm_init)[0] : (batch_size, sequence_length, hidden_size) // output ? (sequence_len, input_size) * (input_size, hidden_size) = (sequence_len, hidden_size)
        # self.lstm(x, lstm_init)[1] : (1, batch_size, hidden_size) // hidden ? hidden !
        #print(self.counter)
        y = x
        self.counter += 1
        # Forward output
        x = self.output(x)

        return x, y

def Optimizer(opt, model):
  optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.lstm_learning_rate)
  return optimizer
