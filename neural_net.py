import numpy as np
from collections import OrderedDict


def relu(X):
    return np.where(X < 0, 0, X)


def drelu(X):
    return np.where(X < 0, 0, 1)


def softmax(Y):
    Z = np.max(Y, axis=0)
    result = np.exp(Y - Z)
    result /= np.sum(np.exp(Y - Z), axis=0, keepdims=True)
    return result


def sigmoid(Y):
    return 1 / (1 + np.exp(-Y))


def initialize_weights(layers):
    Ws, bs = [], []

    for a, b in zip(layers[:-1], layers[1:]):
        W = np.random.randn(b, a) * np.sqrt(2 / a)
        b = np.zeros((b, 1), np.float64)
        Ws.append(W)
        bs.append(b)

    return OrderedDict(Ws=Ws, bs=bs)


def forward_pass(X, params, activations):
    Ws, bs = params['Ws'], params['bs']
    As, Zs = [], []
    A = X

    for W, b, activation in zip(Ws, bs, activations):
        Z = W @ A + b
        A = activation(Z)
        As.append(A)
        Zs.append(Z)
        
    cache = {'As': As, 'Zs': Zs}

    return A, cache


def backpropagation(X, Y, params, cache, dactivations):
    As, Zs, Ws = cache['As'], cache['Zs'], params['Ws']
    m = Y.shape[1]
    total_layers = len(As)

    dWs, dbs = [], []
    A, W =  As[-1], Ws[-1]
    previous_A = As[-2] if total_layers > 1 else X

    dZ = 1 / m * (A - Y)
    dW = dZ @ previous_A.T
    db = np.sum(dZ, axis=1, keepdims=True)
    dWs.append(dW)
    dbs.append(db)
    dA = W.T @ dZ

    for i in range(total_layers - 2, -1, -1):
        A, Z, W, dactivation =  As[i], Zs[i], Ws[i], dactivations[i]

        previous_A = As[i-1] if i > 0 else X
        dZ = dactivation(Z) * dA
        dW = dZ @ previous_A.T
        db = np.sum(dZ, axis=1, keepdims=True)
        dWs.append(dW)
        dbs.append(db)
        dA = W.T @ dZ

    return OrderedDict(dWs=dWs[::-1], dbs=dbs[::-1])


def sigmoid_cross_entropy(Y, predictions):
    m = Y.shape[1]
    result = Y @ np.log(predictions).T + (1-Y) @ np.log(1-predictions).T
    result *= -1/m
    return result[0, 0]


def softmax_cross_entropy(Y, predictions):
    m = Y.shape[1]
    result = Y * np.log(predictions)
    result = np.sum(result)
    result *= -1/m
    return result


def update_weights(params, grads, learning_rate):
    """
    Since params and grads are Ordered Dicts we don't need to reference each
    entry by name.
    """

    for (param_k, param_v), grad_v in zip(params.items(), grads.values()):
        
        new_param_v = []
        for param_entry, grad_entry in zip(param_v, grad_v):
            new_param_v.append(param_entry - learning_rate * grad_entry)
        
        params[param_k] = new_param_v
        
    return params