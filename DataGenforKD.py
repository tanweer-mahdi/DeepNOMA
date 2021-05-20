import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
import torch.nn.functional as F

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


# creating models
ensembles = 4
models = []
for i in range(ensembles):
    models.append(DeepNOMA(structure))

# loading model parameters
for i in range(ensembles):
    name = 'model_'+ str(i) + '.pth'
    load_path = name
    models[i].load_state_dict(torch.load(load_path))
    models[i].eval()

# training with same dataset
dataset = np.load('training_set_0.npy')
tlabel = np.load('labels_0.npy') # true labels

# generating model labels


# creating dataset class for generating labels
class KData(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return self.dataset.shape[1]

    def __getitem__(self, idx):
        return self.dataset[:, idx], self.label[:, idx]

kdataset = KData(dataset, tlabel)
batch = 500
KDataloader = DataLoader(kdataset, batch_size= batch, shuffle= False)
sigmoid = nn.Sigmoid()
temperature = 3;
tensor_list = []
mlabel = np.zeros((tlabel.shape[0], tlabel.shape[1]))
for ind, i in enumerate(KDataloader):
    data, label = i
    ypred = torch.zeros(batch, 60)
    for model in models:
        ypred += model(data)

    ypred /= ensembles
    # ypred = sigmoid(ypred/temperature)
    mlabel[:, ind*batch : (ind+1)*batch] = ypred.detach().numpy().T

with open('mlabels_logit.npy','wb') as file:
    np.save(file,mlabel)


