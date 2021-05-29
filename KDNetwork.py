import torch.utils.data
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pdb

# loading training dataset
training = np.load('training_set_0.npy')
labels = np.load('labels_0.npy')

# loading model labels
mlabels = np.load('mlabels_temp6.npy')


class Tdata(Dataset):
    def __init__(self, training, labels, mlabels):
        self.training = training
        self.labels = labels
        self.mlabels = mlabels

    def __len__(self):
        return training.shape[1]

    def __getitem__(self, idx):
        return training[:, idx], labels[:, idx], mlabels[:, idx]


dataset = Tdata(training, labels, mlabels)

# define the network structure in a dictionary
structure = {
    'input_layer': [280, 250],
    'hidden_1': [250, 250],
    'hidden_2': [250, 250],
    'hidden_3': [250, 250],
    'hidden_4': [250, 250],
    'hidden_5': [250, 250],
    'output_layer': [250, 60],
}


class DeepNOMA(nn.Module):

    def __init__(self, structure):
        super(DeepNOMA, self).__init__()
        # Constructing the Deep Neural Network
        self.hidden = nn.ModuleList()
        for i in structure:
            if i != 'output_layer':
                a, b = structure[i]
                self.hidden.append(nn.Sequential(nn.Linear(a, b), nn.BatchNorm1d(b)))
                # self.hidden.append(nn.Linear(a,b))
                # self.hidden.append(nn.BatchNorm1d(b)) # batch normalization
            else:
                a, b = structure[i]
                self.hidden.append(nn.Linear(a, b))

        # Initializing the DNN
        for i in self.hidden:
            if isinstance(i, nn.Linear):
                nn.init.kaiming_uniform_(i.weight, nonlinearity='relu')
            elif isinstance(i, nn.BatchNorm1d):
                nn.init.constant_(i.weight, 1)
                nn.init.constant_(i.bias, 0)

        # Regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, training):
        L = len(self.hidden)
        for ind, layer in enumerate(self.hidden):
            if ind == 0:
                X = layer(training)  # affine transformation
            elif ind == L - 1:
                X = layer(X)
            else:
                X = layer(X)
                X = F.relu_(X)
                X = self.dropout(X)

        return X


# Training the DNN
model = DeepNOMA(structure)
lossfun = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.0005, amsgrad=False, weight_decay=0)
writer = SummaryWriter('DeepNOMA')
cv = KFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(cv.split(dataset)):

    # creating the sampler
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, batch_size=250, sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size=250, sampler=test_sampler)

    running_loss = 0
    if fold == 0:
        for epoch in range(300):
            for i, val in enumerate(trainloader):
                inputs, targets, mtargets = val
                # clear the gradients
                optimizer.zero_grad()
                # model output
                yhat = model(inputs)
                # calculate loss
                loss = 0.92 * lossfun(yhat/6, mtargets) + 0.08 * lossfun(yhat, targets)
                # backprop

                loss.backward()
                # update model parameter
                optimizer.step()
                # loggin training performance
                running_loss += loss.item()

                if i % 20 == 19:
                    # # calculating validation loss
                    model.eval()
                    validation_loss = 0
                    for j, batch in enumerate(testloader):
                        test, test_labels, _ = batch
                        ypred = model(test)
                        runval = lossfun(ypred, test_labels)
                        validation_loss += runval.item()

                    writer.add_scalars('Training/Validation Loss',
                                       {'Training loss': running_loss / 20, 'Validation Loss': validation_loss / j},
                                       epoch * len(trainloader) + i)

                    model.train()
                    print(running_loss / 20, validation_loss / j, fold)
                    running_loss = 0

# Saving the entire model
save_path = 'kd_model.pth'
torch.save(model.state_dict(), save_path)