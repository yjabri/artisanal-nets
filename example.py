import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import neural_net as nn
from utils import one_hot_encoding
from time import time

mnist = fetch_mldata('MNIST original')

data = mnist['data']
targets = mnist['target']

(train_X_preprocessed, 
 test_X_preprocessed, 
 train_Y_preprocessed, 
 test_Y_preprocessed
) = train_test_split(data, targets, test_size=0.2, random_state=412)

# Transpose data so that each column corresponds to an entry
train_X = train_X_preprocessed.T
test_X = test_X_preprocessed.T

# 0 - 1 scale
train_X = train_X / 255
test_X = test_X / 255

# Preprocess Data - Apply 1 hot encoding to Y
# We also take the transpose so that each column corresponds to an entry
train_Y = one_hot_encoding(train_Y_preprocessed.reshape(1, -1), 10)
test_Y = one_hot_encoding(test_Y_preprocessed.reshape(1, -1), 10)

errors = []

units = [784, 200, 200, 200, 10]
activations = [nn.relu, nn.relu, nn.relu, nn.softmax]
dactivations =  [nn.drelu, nn.drelu, nn.drelu]

batch_size = 128

np.random.seed(217)
params = nn.initialize_weights(units)
total_time = 0

t = 1
epochs = 20
every = 1

print('          epoch|          train|           test|   average time|')
for epoch in range(epochs):
    start = time()
    
    error = 0
    m = train_X.shape[1]
    random_index = np.random.permutation(m)
    
    # Want to do a full pass of the data.
    for i in range(m // batch_size + 1):
        
        idx = slice(i*batch_size, (i+1)*batch_size)

        X_sample = train_X[:, random_index[idx]]
        Y_sample = train_Y[:, random_index[idx]]

        predictions, cache = nn.forward_pass(X_sample, params, activations)
        error = error + nn.softmax_cross_entropy(Y_sample, predictions)

        grads = nn.backpropagation(X_sample, Y_sample, params, cache, dactivations)
        params = nn.update_weights(params, grads, 10**-2)
        
        t = t + 1
        
    elapsed_time = time() - start
    total_time += elapsed_time

    if epoch % every == 0 and epoch > 0:
        
        train_predictions, _ = nn.forward_pass(train_X, params, activations)
        train_accuracy = accuracy_score(np.argmax(train_Y, axis=0), 
                                        np.argmax(train_predictions, axis=0))
        test_predictions, _, = nn.forward_pass(test_X, params, activations)
        test_accuracy = accuracy_score(np.argmax(test_Y, axis=0), 
                                       np.argmax(test_predictions, axis=0)) 
        
        # print('epoch', epoch, end='\n', flush=True)
        print('{:15d}|{:15f}|{:15f}|{:15f}|'.format(
            epoch, train_accuracy, test_accuracy, total_time / every))
        
        total_time = 0
        errors.append(error)