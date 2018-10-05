import numpy as np
import pickle

K = 15
N = 2

with open('expList_v1.1.txt') as fr:
    glist = []
    for line in fr:
        glist.append(line[:-1])

with open('brca_gene_list.txt') as fr:
    tlist = []
    for line in fr:
        tlist.append(line[:-1])

alist = [i for i in tlist if i in glist]

with open('genes_with_exp.txt','w') as fw:
    for gene in alist:
        fw.write(gene+'\n')

atlist = [tlist.index(i) for i in alist]

with open('brca_nbors', 'rb') as fp:
    top = pickle.load(fp)

x_1_8 = np.load('x_1_8.npy')
x_9 = np.load('x_9.npy')
x_10_11 = np.load('x_10_11.npy')
x_12 = np.load('x_12.npy')

Xl = np.concatenate((x_1_8,x_9),axis=1)
Xl = np.concatenate((Xl,x_10_11),axis=1)
Xl = np.concatenate((Xl,x_12),axis=1)

X = Xl[atlist]
gNum,_ = X.shape
# np.save('X',X)

X_cnn = np.zeros((gNum,N*K,X.shape[1]))

for i in range(gNum):
    for j in range(K):
        X_cnn[i,j*N,:] = X[i]
        X_cnn[i,j*N+1,:] = X[top[i][j]]

np.save('X_cnn_{0}_new'.format(K),X_cnn)

with open('benchmark_ind', 'rb') as fp:
    Bench = pickle.load(fp)

dataset = {}
for j in range(5):
    dataset[j] = {}

dg = Bench[0]
for j in range(5):
    dgn = Bench[j+1]
    X = np.concatenate((X_cnn[dg],X_cnn[dgn]),axis=0)
    dataset[j]['X'] = X

dataset['labels'] = np.concatenate((np.ones(len(dg)),np.zeros(len(dgn))))

with open('brca_{0}_new'.format(K), 'wb') as fp:
    pickle.dump(dataset, fp)
