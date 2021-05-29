import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
import torch.nn.functional as F

# loading data
training = 'real_training_set_0.npy'
labels = 'real_labels_0.npy'
# exploring data
train_exp = np.load(training, mmap_mode= 'r')
labels_exp = np.load(labels, mmap_mode = 'r')
input_features = train_exp.shape[0]
num_labels = labels_exp.shape[0]
print("Number of Features:", train_exp.shape[0])
print("Number of Training Samples:", train_exp.shape[1])
print("Number of labels:", num_labels)
# define the network structure in a dictionary
il = input_features
hl = int(input_features*0.1)
structure = {
    'input_layer' : [il , hl],
    'hidden_1' : [hl, hl],
    'hidden_2' : [hl, hl],
    'hidden_3' : [hl, hl],
    'hidden_4' : [hl, hl],
    'hidden_5' : [hl, hl],
    'output_layer': [hl, num_labels],
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
ensembles = 1
models = []
for i in range(ensembles):
    models.append(DeepNOMA(structure))

# loading model parameters
for i in range(ensembles):
    name = 'rmodel_'+ str(i) + '.pth'
    load_path = name
    models[i].load_state_dict(torch.load(load_path))
    models[i].eval()

# loading dev set
dev_data = np.load('rdev_data.npy')
dev_label = np.load('rev_labels.npy')

# # create a dataset
class dev_dataset(Dataset):
    def __init__(self, dev_data, dev_label):
        self.dev_data = dev_data
        self.dev_labels = dev_label

    def __len__(self):
        return self.dev_data.shape[1]

    def __getitem__(self, idx):
        return self.dev_data[:, idx], self.dev_labels[:, idx]

batch = 3
dev_dataset = dev_dataset(dev_data, dev_label)
dev_dataloader = DataLoader(dataset= dev_dataset, batch_size= batch)
total_error = 0

for i in dev_dataloader:
    data, labels = i
    ypred = torch.zeros(batch, num_labels)

    for model in models:
        ypred += model(data)

    #pdb.set_trace()
    ypred /= len(models) # averaging the logits
    ypred[ypred >= 0] = 1 # essentially doing sigmoids
    ypred[ypred < 0] = 0 # essentially doing sigmoids
    temp = torch.abs(ypred - labels)
    total_error += torch.sum(temp).item()

# calculating activity error rate
total_RA = dev_data.shape[1]
aer = total_error / (total_RA*3)
print("Total Activity Error:", aer)



