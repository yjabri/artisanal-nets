import unittest
import numpy as np
from utils import (flatten_params, 
                   unflatten_params, 
                   grad_check, 
                   one_hot_encoding)
import neural_net as nn


class Tests(unittest.TestCase):

    def test_initialize_weights(self):
        layers = [4, 6, 1]
        params = nn.initialize_weights(layers)
        Ws, bs = params['Ws'], params['bs']

        self.assertEqual(len(Ws), 2)
        self.assertEqual(len(bs), 2)
        self.assertEqual(Ws[0].shape, (6, 4))
        self.assertEqual(Ws[1].shape, (1, 6))
        self.assertEqual(bs[0].shape, (6, 1))
        self.assertEqual(bs[1].shape, (1, 1))


    def test_flatten_and_unflatten_params(self):
        layers = [4, 6, 1]
        params = nn.initialize_weights(layers)
        flattened_params = flatten_params(params)
        self.assertEqual(flattened_params.shape, (6 * 4 + 1 * 6 + 6 + 1, ))

        meta = [
            {'name': 'Ws', 'shapes': [(6, 4), (1, 6)]}, 
            {'name': 'bs', 'shapes': [(6, 1), (1, 1)]}]

        reconstructed_params = unflatten_params(flattened_params, meta)
        np.testing.assert_equal(reconstructed_params, params)

    def test_grad_check(self):
        
        def fn(x):
            return np.sum(x ** 3)
        
        x = np.random.randn(10, 1)
        dx = 3 * (x ** 2)

        np.testing.assert_array_almost_equal(dx, grad_check(fn, x))

    def test_one_layer_sigmoid_backpropagation(self):

        m = 10
        X, Y = np.random.randn(4, m), np.random.randn(1, m)
        layers = [4, 1]
        activations = [nn.sigmoid]
        dactivations = []

        params = nn.initialize_weights(layers)
        _, cache = nn.forward_pass(X, params, activations)
        grads = nn.backpropagation(X, Y, params, cache, dactivations)
        flattened_grads = flatten_params(grads)

        params_meta = [
            {'name': 'Ws', 'shapes': [(1, 4)]}, 
            {'name': 'bs', 'shapes': [(1, 1)]}]

        def fn(flattened_params):
            params = unflatten_params(flattened_params, params_meta)
            predictions, _ = nn.forward_pass(X, params, activations)
            return nn.sigmoid_cross_entropy(Y, predictions)


        flattened_params = flatten_params(params)
        flattened_manual_grads = grad_check(fn, flattened_params)

        np.testing.assert_almost_equal(flattened_grads, flattened_manual_grads)

    def test_two_layer_sigmoid_backpropagation(self):

        m = 10
        X, Y = np.random.randn(4, m), np.random.randn(1, m)
        layers = [4, 6, 1]
        activations = [nn.relu, nn.sigmoid]
        dactivations = [nn.drelu]

        params = nn.initialize_weights(layers)
        _, cache = nn.forward_pass(X, params, activations)
        grads = nn.backpropagation(X, Y, params, cache, dactivations)
        flattened_grads = flatten_params(grads)

        params_meta = [
            {'name': 'Ws', 'shapes': [(6, 4), (1, 6)]}, 
            {'name': 'bs', 'shapes': [(6, 1), (1, 1)]}]

        def fn(flattened_params):
            params = unflatten_params(flattened_params, params_meta)
            predictions, _ = nn.forward_pass(X, params, activations)
            return nn.sigmoid_cross_entropy(Y, predictions)


        flattened_params = flatten_params(params)
        flattened_manual_grads = grad_check(fn, flattened_params)

        np.testing.assert_almost_equal(flattened_grads, 
                                       flattened_manual_grads)

    def test_one_layer_softmax_backpropagation(self):

        m = 10
        X = np.random.randn(4, m)

        Y = np.random.randint(3, size=(1, m))
        Y = one_hot_encoding(Y, 3)
        
        layers = [4, 3]
        activations = [nn.softmax]
        dactivations = []

        params = nn.initialize_weights(layers)
        _, cache = nn.forward_pass(X, params, activations)
        grads = nn.backpropagation(X, Y, params, cache, dactivations)
        flattened_grads = flatten_params(grads)

        params_meta = [
            {'name': 'Ws', 'shapes': [(3, 4)]}, 
            {'name': 'bs', 'shapes': [(3, 1)]}]

        def fn(flattened_params):
            params = unflatten_params(flattened_params, params_meta)
            predictions, _ = nn.forward_pass(X, params, activations)
            return nn.softmax_cross_entropy(Y, predictions)


        flattened_params = flatten_params(params)
        flattened_manual_grads = grad_check(fn, flattened_params)

        np.testing.assert_almost_equal(flattened_grads, 
                                       flattened_manual_grads)


    def test_two_layer_softmax_backpropagation(self):

        m = 10
        X = np.random.randn(4, m)

        Y = np.random.randint(3, size=(1, m))
        Y = one_hot_encoding(Y, 3)

        layers = [4, 6, 3]
        activations = [nn.relu, nn.softmax]
        dactivations = [nn.drelu]

        params = nn.initialize_weights(layers)
        _, cache = nn.forward_pass(X, params, activations)
        grads = nn.backpropagation(X, Y, params, cache, dactivations)
        flattened_grads = flatten_params(grads)

        params_meta = [
            {'name': 'Ws', 'shapes': [(6, 4), (3, 6)]}, 
            {'name': 'bs', 'shapes': [(6, 1), (3, 1)]}]

        def fn(flattened_params):
            params = unflatten_params(flattened_params, params_meta)
            predictions, _ = nn.forward_pass(X, params, activations)
            return nn.softmax_cross_entropy(Y, predictions)


        flattened_params = flatten_params(params)
        flattened_manual_grads = grad_check(fn, flattened_params)

        np.testing.assert_almost_equal(flattened_grads, 
                                       flattened_manual_grads)


if __name__ == '__main__':
    unittest.main()
