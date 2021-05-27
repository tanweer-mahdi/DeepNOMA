import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import pdb
# number of models in the ensemble
ensemble = 5
P = 125000  # number of samples
dev_sample = 40000 # number of dev samples
# create a dummy system model
N = 60  # number of UE
M = 40  # number of subcarrier
distances = np.random.rand(N)*0.2 # cell radius is 200 meter
noisevar = 1e-11 # noise variance
J = 9 # number of slots
pa = 0.05 # activation ratio
tp = 20 # transmit power in dBm
codebook = 'sparse' # choice of codebook

if codebook == 'sparse':
    phi = np.zeros((2*M, N))
    nz_entries = int(M/2)  # number of non-zero entries
    for i in range(phi.shape[1]):
        nz_index = np.random.choice(np.arange(M), size=nz_entries, replace=False)
        phi[nz_index, i] = np.random.normal(0, 1, (nz_entries,))
elif codebook == 'dense':
    phi = np.random.normal(0, 1, (2*M, N))
else:
    pass


# normalizing codebook
vecnorm = 1 / np.linalg.norm(phi, 2, axis=0)
phi = np.dot(phi, np.diag(vecnorm))  # normalizing each sequence to unit norm vectors


# Generate training samples
def gensample(N, M, J, phi, noisevar):
    M = 2*M
    au = int(np.ceil(pa * N))
    # au = np.random.randint(1, int(np.ceil(pa * N)) + 1)
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

    # creating channel vectors
    cv = np.zeros((M*au, J))

    for slot in range(J):
        uchannel = np.zeros((M, au))

        for i, val in enumerate(uset):
            rp = -128.1 - 36.7*np.log10(distances[val])+tp;
            rp = np.power(10, rp/10)
            uchannel[:, i] = np.random.normal(0, 1, (M, ))*np.sqrt(rp)

        cv[:, slot] = np.ravel(uchannel)



    y = np.dot(C, cv)
    y += np.random.normal(0, noisevar, (M, J))

    # vectorization and normalization
    y = np.ravel(y)
    mu = np.mean(y)
    sigma = np.std(y)

    y = (y-mu)/sigma

    return y.astype(dtype='float32'), labels
    #return np.ravel(y).astype(dtype='float32'), labels




for ii in range(ensemble):
    dataset = np.zeros((2* M * J, P), dtype='float32')
    labels = np.zeros((N, P), dtype='float32')

    for j in range(P):
        data, label = gensample(N,M,J,phi,noisevar)
        dataset[:, j] = data
        labels[:, j] = label


    training_name = 'real_training_set_' + str(ii) + '.npy'
    labels_name = 'real_labels_' + str(ii) + '.npy'
    # ## saving training data
    with open(training_name,'wb') as file: # using "with" while opening file is a good idea. It properly closes the file.
        np.save(file, dataset)

    ## saving training labels
    with open(labels_name,'wb') as file:
        np.save(file, labels)



# creating dev data

dataset = np.zeros((2* M * J, dev_sample), dtype='float32')
labels = np.zeros((N, dev_sample), dtype='float32')

for i in range(dev_sample):
    data, label = gensample(N,M,J,phi,noisevar)
    dataset[:, i] = data
    labels[:, i] = label

# saving dev data
with open('real_dev_data.npy', 'wb') as file:
    np.save(file, dataset)

with open('real_dev_labels.npy', 'wb') as file:
    np.save(file, labels)
