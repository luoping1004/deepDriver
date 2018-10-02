import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
import pickle
import numpy as np
import time
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True)
from sklearn.metrics import roc_auc_score

start = time.time()
# 14.45 min

# load data
K = 7
F = 24
filter_size = 2
D = 48
R = 5
print('K =', K)
filename = 'coad_{0}_new'.format(K)
with open(filename, 'rb') as fp:
    dataset = pickle.load(fp)

input_shape = (dataset[0]['X'].shape[1],dataset[0]['X'].shape[2])

model = Sequential()
model.add(Conv1D(F, filter_size, strides=2, padding='valid',
                 activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(2))
model.add(Conv1D(F, filter_size, padding='same',activation='relu'))
model.add(MaxPooling1D(2))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(D, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer='Adam')

y = dataset['labels']
y_scores = np.zeros((5 * R, y.shape[0]))

for j in range(5):
    X = dataset[j]['X']

    for r in range(R):
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train, batch_size=8,epochs=10,verbose=0)
            y_scores[j*R+r,test_index] = model.predict_proba(X_test).reshape(-1)

y_true = dataset['labels']
for i in range(5*R-1):
    y_true = np.concatenate((y_true,dataset['labels']))
y_score = y_scores.reshape(-1)

print(roc_auc_score(y_true, y_score))

np.save('y_true_deep_10fold',y_true)
np.save('y_score_deep_10fold',y_score)

end = time.time()
print('time:',(end-start)/60, 'min')
