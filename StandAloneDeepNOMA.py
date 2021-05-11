import torch.utils.data
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
# loading data
training = 'training_set_0.npy'
labels = 'labels_0.npy'

# exploring data
train_exp = np.load(training, mmap_mode= 'r')
print("Number of Features:", train_exp.shape[0])
print("Number of Training Samples:", train_exp.shape[1])

# creating dataset
class nomadata(Dataset):
    def __init__(self, training, labels):
        self.mmapped = np.load(training)
        #self.mmapped = np.load(training, mmap_mode = 'r')
        self.labels = np.load(labels)

    def __len__(self):
        return self.mmapped.shape[1]

    def __getitem__(self,idx):
        sample = [self.mmapped[:,idx], self.labels[:,idx]]
        return sample


nomadata = nomadata(training,labels)


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



# Codes for visualizing logs
writer = SummaryWriter('DeepNOMA')
model = DeepNOMA(structure)
nomadata_loader = DataLoader(nomadata, batch_size= 20)
writer.add_graph(model, next(iter(nomadata_loader))[0])
writer.close()


# Training the DNN
lossfun = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr = 0.0005, amsgrad=False, weight_decay= 0)
cv = KFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(cv.split(nomadata)):

    # creating the sampler
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    trainloader = DataLoader(nomadata, batch_size = 250, sampler  = train_sampler)
    testloader = DataLoader(nomadata, batch_size= 250, sampler = test_sampler)

    # resetting the parameters
    # model = DeepNOMA(structure)
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


    running_loss = 0
    if fold == 0:
        for epoch in range(200):
            for i, val in enumerate(trainloader):
                inputs, targets = val
                # clear the gradients
                optimizer.zero_grad()
                # model output
                yhat = model(inputs)
                # calculate loss
                loss = lossfun(yhat, targets)
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
                        test, labels = batch
                        ypred = model(test)
                        runval = lossfun(ypred, labels)
                        validation_loss += runval.item()

                    writer.add_scalars('Training/Validation Loss', {'Training loss': running_loss/20, 'Validation Loss': validation_loss/j}, epoch*len(trainloader) + i)

                    model.train()
                    print(running_loss/20, validation_loss/j, fold)
                    running_loss = 0



    # evaluating model for current fold



# Saving the entire model
save_path = 'model.pth'
torch.save(model.state_dict(), save_path)