import numpy as np
from collections import OrderedDict
from numpy import ndarray
from typing import List


def one_hot_encoding(X, classes):
    """This implementation is explicit about the number of classes"""
    base = np.zeros((classes, X.shape[1]))

    for i, entry in enumerate(X[0, :]):
        base[int(entry), i] = 1

    return base


def flatten_params(params: OrderedDict) -> ndarray:

    flattened_values = []

    for values in params.values():
        for value in values:
            flattened_value = np.concatenate(value)
            flattened_values.append(flattened_value)

    return np.concatenate(flattened_values)


def unflatten_params(values: ndarray, meta: List[dict]) -> OrderedDict:
    """
    meta - A list of dictionaries of that each contain the keys `name` and
    `shapes`. For example if the orginal `params` used to construct `values`
    was `Ws` and `bs` respetively where `Ws` has 2 entries of shape `(6, 4)`
    and `(1, 4)` and `bs` has 2 entries of shape `(6, 1)`  and `(1, 1)` then
    `meta` would be
        [{
            'name': 'Ws',
            'shapes': [(6, 4), (1, 4)]
        }, {
            'name': 'bs',
            'shapes': [(6, 1), (1, 1)]
        }]
    the order of the entries in `meta` should correspond to the order in the
    OrderedDict `params`.
    """

    index = 0
    params = OrderedDict()

    for entry in meta:

        name = entry['name']
        shapes = entry['shapes']

        param_entries = []
        for (n, m) in shapes:
            k = n * m
            param_entry = values[index:index+k].reshape(n, m)
            param_entries.append(param_entry)
            index = index + k

        params[name] = param_entries

    return params


def grad_check(fn, params, epsilon=10**-7):

    partials = np.zeros_like(params)

    for i, _ in enumerate(partials):
        bump = np.zeros_like(params, dtype=np.float64)
        bump[i] = epsilon

        partial = fn(params + bump) - fn(params - bump)
        partial /= 2 * epsilon
        partials[i] = partial

    return partials
