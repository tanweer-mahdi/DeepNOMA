import numpy as np
from sklearn.model_selection import train_test_split
import pdb
# create a dummy system model
N = 60;  # number of UE
M = 40;  # number of subcarrier
# phi = np.random.normal(0, 1, (M, N))
# vecnorm = 1 / np.linalg.norm(phi, 2, axis=0)
# phi = np.dot(phi, np.diag(vecnorm))  # normalizing each sequence to unit norm vectors
# creating sparse codes
phi = np.zeros((M,N))
nz_entries = 10 # number of non-zero entries
for i in range(phi.shape[1]):
    nz_index = np.random.choice(np.arange(M), size = nz_entries, replace = False)
    phi[nz_index, i] = np.random.normal(0, 1, (nz_entries,))


noisevar = 1e-11
J = 7
pa = 0.05


# Generate training samples
def gensample(N, M, J, phi, noisevar):
    au = int(np.ceil(pa * N))
    uset = np.random.choice(np.arange(N), size=au, replace=False)
    labels = np.zeros(N,dtype='float32')
    labels[uset] = 1
    # creating the combined dictionary
    C = np.diag(phi[:, uset[0]])
    for i in uset[1:]:
        C = np.hstack((C, np.diag(phi[:, i])))

    # creating channel vector. Channel is changing over subcarriers but not over timeslot
    # cv = np.random.normal(0, 1, (M * au, 1))
    # t= cv
    # for i in range(J-1):
    #     cv = np.hstack((cv,t))

    cv = np.random.normal(0, 1, (M*au, J))

    y = np.dot(C, cv)
    y += np.random.normal(0, noisevar, (M, J))

    # vectorization and normalization
    y = np.ravel(y)
    mu = np.mean(y)
    sigma = np.std(y)

    y = (y-mu)/sigma

    return y.astype(dtype='float32'), labels
    #return np.ravel(y).astype(dtype='float32'), labels


P = 125000  # number of samples
dataset = np.zeros((M * J, P), dtype='float32')
labels = np.zeros((N, P), dtype='float32')

for i in range(P):
    data, label = gensample(N,M,J,phi,noisevar)
    dataset[:, i] = data
    labels[:, i] = label


# Training data/ Dev data splitting
train_data, dev_data, train_label, dev_label = train_test_split(dataset.T, labels.T, test_size= 0.20)

# ## saving training data
with open('training_set.npy','wb') as file: # using "with" while opening file is a good idea. It properly closes the file.
    np.save(file,train_data.T)

## saving training labels
with open('labels.npy','wb') as file:
    np.save(file,train_label.T)

# saving dev data
with open('dev_data.npy', 'wb') as file:
    np.save(file, dev_data.T)

with open('dev_labels.npy', 'wb') as file:
    np.save(file, dev_label.T)