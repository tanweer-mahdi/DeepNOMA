import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
import torch.nn.functional as F
# define the network structure in a dictionary
# define the network structure in a dictionary
structure = {
    'input_layer' : [280 , 250],
    'hidden_1' : [250, 250],
    'hidden_2' : [250, 250],
    'hidden_3' : [250, 250],
    'hidden_4' : [250, 250],
    'hidden_5' : [250, 250],
    'output_layer': [250, 60],
}


class DeepNOMA(nn.Module):
    def __init__(self, structure):
        super(DeepNOMA, self).__init__()
        # Constructing the Deep Neural Network
        self.hidden = nn.ModuleList()
        for i in structure:
            if i != 'output_layer':
                a,b = structure[i]
                self.hidden.append(nn.Sequential(nn.Linear(a,b), nn.BatchNorm1d(b)))
                # self.hidden.append(nn.Linear(a,b))
                # self.hidden.append(nn.BatchNorm1d(b)) # batch normalization
            else:
                a,b = structure[i]
                self.hidden.append(nn.Linear(a,b))

        # Initializing the DNN
        for i in self.hidden:
            if isinstance(i, nn.Linear):
                nn.init.kaiming_uniform_(i.weight, nonlinearity= 'relu')
            elif isinstance(i, nn.BatchNorm1d):
                nn.init.constant_(i.weight, 1)
                nn.init.constant_(i.bias, 0)


        # Regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, training):
        L = len(self.hidden)
        for ind,layer in enumerate(self.hidden):
            if ind == 0:
                X = layer(training)  # affine transformation
            elif ind == L - 1:
                X = layer(X)
            else:
                X = layer(X)
                X = F.relu_(X)
                X = self.dropout(X)

        return X


# loading model
model = DeepNOMA(structure)
load_path = 'model.pth'
model.load_state_dict(torch.load(load_path))
model.eval()

# loading dev set
dev_data = np.load('dev_data.npy')
dev_label = np.load('dev_labels.npy')

# create a dataset
class dev_dataset(Dataset):
    def __init__(self, dev_data, dev_label):
        self.dev_data = dev_data
        self.dev_labels = dev_label

    def __len__(self):
        return self.dev_data.shape[1]

    def __getitem__(self, idx):
        return self.dev_data[:, idx], self.dev_labels[:, idx]

dev_dataset = dev_dataset(dev_data, dev_label)
dev_dataloader = DataLoader(dataset= dev_dataset, batch_size= 10)

for i in dev_dataloader:
    data, labels = i
    ypred = model(data)
    print(ypred, labels)
    for j, val in enumerate(labels):
        a = val.numpy()
        ind = np.where(a==1)
        temp = ypred[j, ind]
        print(ind)
        print(temp)
    pdb.set_trace()

