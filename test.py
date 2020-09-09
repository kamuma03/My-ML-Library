# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 08:48:39 2020

@author: mukan
"""
import kaLib
import numpy as np 
np.set_printoptions(suppress=True) # dont print every number in scientific form
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import sys
sys.path.append('..') # add utils location to path
from utils import get_regression_data, visualise_regression_data  # function to create dummy data for regression


def sample_polynomial_data(m=20, order=3, _range=1):
    coeffs = np.random.randn(order + 1) # initialise random coefficients for each order of the input + a constant offset
    print(Polynomial(coeffs))
    poly_func = np.vectorize(Polynomial(coeffs)) # 
    X = np.random.randn(m)
    X = np.random.uniform(low=-_range, high=_range, size=(m,))
    Y = poly_func(X)
    return X, Y, coeffs #returns X (the input), Y (labels) and coefficients for each power

    
# funtion to create the polynominal data from the regression data
def create_polynomial_inputs(X, order=3):
    new_dataset = np.array([np.power(X, i) for i in range(1, order + 1)]).T ## add powers of the original feature to the design matrix
    return new_dataset # new_dataset should be shape [m, order]    
    
# X, Y = get_regression_data() # get dummy regression data


# H = kaLib.LinearHypothesis()

# to test the random search optimizer
# best_weights, best_bias = kaLib.random_search(H, X, Y, 10000) # do 10000 samples in a random search 
# H.update_params(best_weights, best_bias) # make sure to set our model's weights to the best values we found
# kaLib.plot_h_vs_y(X, H(X), Y) # plot model predictions agains labels

# to test the grid search optimizer
# best_weights, best_bias = kaLib.grid_search(H, X, Y)  
# H.update_params(best_weights, best_bias) # make sure to set our model's weights to the best values we found
# kaLib.plot_h_vs_y(X, H(X), Y) # plot model predictions agains labels


# to test gradient discent without batches
# num_epochs = 100
# learning_rate = 0.1
# kaLib.train(num_epochs, X, Y, H, learning_rate, plot_cost_curve=True) # train model and plot cost curve
# visualise_regression_data(X, Y, H(X)) # plot predictions and true data


# to test the gradient discent with the batches
# learning_rate = 0.00001
# m = 10000
# num_updates = 10 * m
# # X, Y = sample_linear_data(m)
# dataset = list(zip(X, Y))
# mini_batch_data_loader = kaLib.create_batches(dataset, batch_size=32)
# kaLib.train_with_batches(num_updates, mini_batch_data_loader, H, learning_rate, plot_cost_curve=True)


# # to test multivariable liners regression
# m = 20 # how many examples do we want?
# order = 3 # how many powers do we want to raise our input data to?
# X, Y, ground_truth_coeffs = sample_polynomial_data(m, order)

# num_epochs = 200
# learning_rate = 0.1
# highest_order_power = 4

# polynomial_augmented_inputs = create_polynomial_inputs(X, highest_order_power) ## need normalization to put higher coefficient variables on the same order of magnitude as the others
# H = kaLib.MultiVariableLinearHypothesis(n_features=highest_order_power) ## initialise multivariate regression model

# kaLib.train(num_epochs, polynomial_augmented_inputs, Y, H, learning_rate) ## train model
# kaLib.plot_h_vs_y(X, H(polynomial_augmented_inputs), Y)



# to test multivariable liners regression with the nesterov momentum
m = 100 # how many examples do we want?
order = 3 # how many powers do we want to raise our input data to?
X, Y, ground_truth_coeffs = sample_polynomial_data(m, order)

num_epochs = 20000
learning_rate = 0.0001
highest_order_power = 5
momentum = 0.9

polynomial_augmented_inputs = create_polynomial_inputs(X, highest_order_power) ## need normalization to put higher coefficient variables on the same order of magnitude as the others
H = kaLib.MultiVariableLinearHypothesis(n_features=highest_order_power) ## initialise multivariate regression model

a = kaLib.train_nesterov_momentum(num_epochs, polynomial_augmented_inputs, Y, H, learning_rate, momentum) ## train model
kaLib.plot_h_vs_y(X, H(polynomial_augmented_inputs), Y)
kaLib.plot_loss(a)

# normal training for  comparison
M = kaLib.MultiVariableLinearHypothesis(n_features=highest_order_power) ## initialise multivariate regression model

b = kaLib.train(num_epochs, polynomial_augmented_inputs, Y, M, learning_rate) ## train model
kaLib.plot_h_vs_y(X, M(polynomial_augmented_inputs), Y)
kaLib.plot_loss(b)