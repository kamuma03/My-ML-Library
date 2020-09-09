import numpy as np
import matplotlib.pyplot as plt
from time import time
from random import shuffle
from sklearn.model_selection import train_test_split

def test_print():
    print("i have finished the compile")
    
# function to plot the loses against the epochs
def plot_loss(losses):
    plt.figure() # make a figure
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses) # plot costs

#plot the data and the predictions
def plot_h_vs_y(X, y_hat, Y):
    plt.figure()
    plt.scatter(X, Y, c='r', label='Label')
    plt.scatter(X, y_hat, c='b', label='Hypothesis', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# Class for the linear regression
class LinearHypothesis:
    def __init__(self): 
        self.w = np.random.randn() ## weight
        self.b = np.random.randn() ## bias
    
    def __call__(self, X): ## how do we calculate output from an input in our model?
        y_hat = self.w*X + self.b ## make linear prediction
        return y_hat
    
    def update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's w to the new w
        self.b = new_b ## set this instance's b to the new b
        
    def calc_deriv(self, X, y_hat, labels):
        m = len(labels) ## m = number of examples
        diffs = y_hat - labels ## calculate errors
        dLdw = 2*np.array(np.sum(diffs*X) / m) ## calculate derivative of loss with respect to weights
        dLdb = 2*np.array(np.sum(diffs)/m) ## calculate derivative of loss with respect to bias
        return dLdw, dLdb ## return rate of change of loss wrt w and wrt b


# Loss function implementation mean_squared_error   
def mse_loss(y_hat, labels): # define our criterion (loss function)
    errors = y_hat - labels ## calculate errors
    squared_errors = np.square(errors) ## square errors
    mean_squared_error = np.sum(squared_errors) / len(y_hat) ## calculate mean 
    return mean_squared_error # return loss

# optimizer random_search
def random_search( H, X, Y, n_samples, limit=20):
    """Try out n_samples of random parameter pairs and return the best ones"""
    best_weights = None ## no best weight found yet
    best_bias = None ## no best bias found yet
    lowest_cost = float('inf') ## initialize it very high (how high can it be?)
    for i in range(0, n_samples): ## try this many different parameterisations
        w = np.random.uniform(-limit, limit) ## randomly sample a weight within the limits of the search
        b = np.random.uniform(-limit, limit) ## randomly sample a bias within the limits of the search
        # print(w, b)
        H.update_params(w, b) ## update our model with these random parameters
        y_hat = H(X) ## make prediction
        cost = mse_loss(y_hat, Y) ## calculate loss
        if cost < lowest_cost: ## if this is the best parameterisation so far
            lowest_cost = cost ## update the lowest running cost to the cost for this parameterisation
            best_weights = w ## get best weights so far from the model
            best_bias = b ## get best bias so far from the model
    return best_weights, best_bias ## return the best weight and best bias

#optimizer grid search
def grid_search( H, X, Y, limit=20, resolution = 0.1):
    """Try out grid parameters pairs and return the best ones"""
    best_weights = None ## no best weight found yet
    best_bias = None ## no best bias found yet
    lowest_cost = float('inf') ## initialize it very high (how high can it be?)
    list_weights = np.arange(-limit, limit, 0.1).tolist()
    list_bias = np.arange(-limit, limit, 0.1).tolist()
    for i in range(0, len(list_weights)):
        for j in range(0, len(list_bias)):## try this many different parameterisations
            w = list_weights[i] 
            b = list_bias[j] 
            # print(w, b)
            H.update_params(w, b) ## update our model with these  parameters
            y_hat = H(X) ## make prediction
            cost = mse_loss(y_hat, Y) ## calculate loss
            if cost < lowest_cost: ## if this is the best parameterisation so far
                lowest_cost = cost ## update the lowest running cost to the cost for this parameterisation
                best_weights = w ## get best weights so far from the model
                best_bias = b ## get best bias so far from the model
    return best_weights, best_bias ## return the best weight and best bias


# function to do training for all the input toghether 
def train(num_epochs, X, Y, H, learning_rate):
    all_costs = [] ## initialise empty list of costs to plot later
    for e in range(num_epochs): ## for this many complete runs through the dataset
        y_hat = H(X) ## make predictions
        cost = mse_loss(y_hat, Y) ## compute loss 
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) ## calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw ## compute new model weight using gradient descent update rule
        new_b = H.b - learning_rate * dLdb ## compute new model bias using gradient descent update rule
        H.update_params(new_w, new_b) ## update model weight and bias
        all_costs.append(cost) ## add cost for this batch of examples to the list of costs (for plotting)
    return all_costs
    
 # create batches, the inut dataset should be a zipped list of X and Y
def create_batches(dataset, batch_size=4):
    shuffle(dataset) # shuffle the dataset. why?
    idx = 0 # initialise starting point in dataset (index of first example to be put into the next batch)
    batches = []
    while idx < len(dataset): # while starting point index is less than the length of the dataset 
        if idx + batch_size < len(dataset): # if enough examples remain to make a whole batch
            batch = dataset[idx: idx + batch_size] # make a batch from those examples 
        else: # otherwise
            batch = dataset[idx:] # take however many examples remain (less than batch size)
        batches.append(batch) # add this batch to the list of batches
        idx += batch_size # increment the starting point for the next batch
    batches = [np.array(list(zip(*b))) for b in batches] # unzip the batches into lists of inputs and outputs so batch = [all_inputs, all_outputs] rather than batch = [(input_1, output_1), ..., (input_batch_size, output_batch_size)]
    return batches


# training with batches 
def train_with_batches(num_updates, data_loader, H, learning_rate):
    costs = [] # initialise empty list of costs to plot later
    update_idx = 0
    inference_times = []
    update_times = []
    while update_idx < num_updates: # for this many complete runs through the dataset
        batch_costs = []
        for x, y in data_loader:
            inference_start = time() # get time at start of inference
            y_hat = H(x) # make predictions
            inference_times.append(time() - inference_start) # add duration of inference
            cost = mse_loss(y_hat, y) # compute loss 
            update_start = time()
            dLdw, dLdb = H.calc_deriv(x, y_hat, y) # calculate gradient of current loss with respect to model parameters
            new_w = H.w - learning_rate * dLdw # compute new model weight using gradient descent update rule
            new_b = H.b - learning_rate * dLdb # compute new model bias using gradient descent update rule
            H.update_params(new_w, new_b) # update model weight and bias
            update_times.append(time() - update_start)
            update_idx += 1
            batch_costs.append(cost)
            #prop_complete = round((update_idx / num_updates) * 100)     
            #print('\r' + ["|", "/", "-", "\\"][update_idx % 4], end='')
            #print(f'\r[{prop_complete * "=" + (0 - prop_complete) * "-"}]', end='')
        costs.append(np.mean(batch_costs)) # add cost for this batch of examples to the list of costs (for plotting)
        return costs
    
    
# function to test the model
def test(X, Y, H):
    y_hat = H(X) ## make predictions
    loss = np.sum((y_hat - Y)**2) / len(Y) # calculate mean squared error
    return loss

# multivariable linear regression class with regolarizzation factor
class MultiVariableLinearHypothesis:
    def __init__(self, n_features, regularisation_factor =1): ## add regularisation factor as parameter
        self.n_features = n_features
        self.regularisation_factor = regularisation_factor ## add self.regularisation factor
        self.b = np.random.randn()
        self.w = np.random.randn(n_features)
        
    def __call__(self, X): # what happens when we call our model, input is of shape (n_examples, n_features)
        y_hat = np.matmul(X, self.w) + self.b # make prediction, now using vector of weights rather than a single value
        return y_hat # output is of shape (n_examples, 1)
    
    def update_params(self, new_w, new_b):
        self.w = new_w
        self.b = new_b
        
    def calc_deriv(self, X, y_hat, labels):
        diffs = y_hat-labels
        dLdw = 2 * np.array([np.sum(diffs * X[:, i]) / len(labels) for i in range(self.n_features)]) 
        dLdw += 2 * self.regularisation_factor * self.w ## add regularisation term gradient
        dLdb = 2 * np.sum(diffs) / len(labels)
        return dLdw, dLdb

# nomralizzation of the data
def standardize_data(dataset):
    mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0) ## get mean and standard deviation of dataset
    standardized_dataset  = (dataset-mean)/std
    return standardized_dataset

# split data in train, test and if required validation as well
def split_dataset(train_data, val_data=False, test_size=0.25, val_size=None, shuffle=True):
    if val_data==False:
        train_data, test_data = train_test_split(train_data, test_size, shuffle)
        return train_data, test_data
    else:
       train_data, test_data = train_test_split(train_data, test_size+val_size, shuffle)
       val_size = val_size/(test_size+val_size)
       test_data, val_data = train_test_split(test_data, val_size, shuffle)
       return train_data, test_data, val_data

##  Train with the gradient dischent with the nesterov momentum
## https://github.com/keras-team/keras/issues/966
## https://arxiv.org/pdf/1212.0901v2.pdf 
def train_nesterov_momentum(num_epochs, X, Y, H, learning_rate, momentum = 0.9):
    all_costs = [] ## initialise empty list of costs to plot later
    vtw_1 = 0 ## initialize the previous velocity
    vtb_1 = 0
    vtw = 0 ## initialize the current velocity
    vtb = 0  
    for e in range(num_epochs): ## for this many complete runs through the dataset
        y_hat = H(X) ## make predictions
        cost = mse_loss(y_hat, Y) ## compute loss 
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) ## calculate gradient of current loss with respect to model parameters
        vtw = momentum*vtw_1 - learning_rate * dLdw ## calculate the current velocity
        vtb = momentum*vtb_1 - learning_rate * dLdb ## calculate the current velocity
        new_w = H.w + momentum * momentum * vtw_1 - (1 + momentum) * learning_rate * dLdw  ## compute new model weight using gradient descent with nesterov momentum update rule
        new_b = H.b + momentum * momentum * vtb_1 - (1 + momentum) * learning_rate * dLdb  ## compute new model bias using gradient descent with nesterov momentum update rule
        H.update_params(new_w, new_b) ## update model weight and bias
        all_costs.append(cost) ## add cost for this batch of examples to the list of costs (for plotting)
        vtw_1 = vtw
        vtb_1 = vtb
    return all_costs       
        
        
        
        



