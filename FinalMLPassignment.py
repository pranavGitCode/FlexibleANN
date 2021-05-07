# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:25:51 2021

Full Project Final Code
"""
from random import seed
from random import randrange, random, shuffle
from math import exp
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from seaborn import heatmap
mScaler = MinMaxScaler()
lEn = LabelEncoder()


"""Choose dataset to run MLP by assigning numerical index of data eg. choose_data = 1 for 'Iris'"""
dataset_name_list = ['Iris', 'Breast cancer', 'Banknote', 'Vowel Data']
choose_data = 4
data_name = dataset_name_list[choose_data-1]
if (choose_data == 1):
    from sklearn.datasets import load_iris
    dataset = load_iris(as_frame = True)
    X_iris = dataset.data
    y_iris = dataset.target
    dataset = X_iris.join(y_iris)
elif (choose_data == 2):
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer(as_frame = True)
    X_cancer = dataset.data
    y_cancer = dataset.target
    dataset = X_cancer.join(y_cancer)
elif (choose_data == 3):
    dataset = pd.read_csv("data_banknote_authentication.txt")
else:
    dataset = pd.read_csv("data.csv", header = None) 

""""Preprocessing : Assuming data is all numerical MinMaxScaling of feature variables, Target variable is LabelEncoded"""
dataset.iloc[:,:-1] = mScaler.fit_transform(dataset.iloc[:,:-1])
dataset.iloc[:,-1] = lEn.fit_transform(dataset.iloc[:,-1])
train_set, test_set = train_test_split(dataset, test_size = 0.2, random_state = 6, shuffle = True)

def initialize_network(n_inputs, n_hidden, n_outputs):
    """Initialize a network"""
    network = list()
    hidden_layer = [{'weights' : [random() for j in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for j in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def initialize_activationFn(k_order):                     
    """ Order of Coeff f(x) = k0 + k1*x + k2*(x)^2 , variable """
    coeff = {'values': [random() for i in range(k_order)], 'update' :[0 for i in range(k_order)]}
    return coeff
    

def activate(weights, inputs):
    """Activation of a neuron for an input """
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
        # print(i)
    return activation

 
def transfer(activation, coeff):
    """Current order 2: return (coeff[0] + coeff[1]*activation + coeff[2]*(activation)**2)"""
    return (coeff['values'][0] + coeff['values'][1]*activation)

def transfer_derivative(activation, coeff):
        return (coeff['values'][1])

def softmax(x):
    """Compute softmax values"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def forward_propagate(network, coeff,  valuesIn ):
    """ Forward propagate input to network output: last layer - Softmax(z) activation function"""
    inputs = valuesIn
    for layer in network[:-1]:
        # print(1)
        new_inputs = []
        for neuron in layer:
            neuron['activation'] =  activate(neuron['weights'], inputs)
            neuron['output'] = transfer(neuron['activation'], coeff)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    #Separately handles last layer
    new_inputs = []
    for neuron in network[-1]:
        neuron['activation'] =  activate(neuron['weights'], inputs)
        new_inputs.append(neuron['activation'])
    inputs = list(softmax(new_inputs))
    for i in range(len(inputs)):
        network[-1][i]['output'] = inputs[i]
    # print(network)
    return inputs


def backward_propagate_error(network, coeff, expected):
    """Backpropagate error and store in neurons"""
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != (len(network)-1):
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
            coeff1 = []
            coeff2 = []
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['activation'], coeff)
                # if (neuron['delta'] < 0):
                #     print("Neuron error %.8f in Layer %d",(neuron['delta'], j))
                coeff1.append(errors[j]*1)
                coeff2.append(errors[j]*neuron['activation'])
            coeff['update'][0] = np.average(coeff1)
            coeff['update'][1] = np.average(coeff2)

        else:
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = neuron['output'] - expected[j]
                # if (neuron['delta'] < 0):
                #     print("Neuron error %.8f in Layer %d",(neuron['delta'], j))

def update_weights(network, coeff, row, l_rate):
    """Update coeff of g(x) = k1 + k2*x  and neuron weights"""
    for i in range(len(coeff['values'])):
        coeff['values'][i] -= l_rate * coeff['update'][i]
    for i in range(len(network)):
        inputs = row[:-1]                
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                temp = l_rate * neuron['delta'] * inputs[j] 
                neuron['weights'][j] -= temp
                # print("neuron weight{} \n".format(neuron['weights'][j]))
            temp = l_rate * neuron['delta']
            neuron['weights'][-1] -= temp

def train_network(network, coeff, train, test, l_rate, n_epoch, n_outputs):
    metric_values = {'train_accuracy':list(), 'test_accuracy':list(), 'train_loss' :list(), 'test_loss':list(), 'param': list()}
    actual_train = [row[-1] for row in train]
    actual_test = [row[-1] for row in test]
    for epoch in range(n_epoch):
        sum_error_train = 0
        sum_error_test = 0
        predict_train = []
        predict_test = []       
        for row in train:
            outputs = forward_propagate(network, coeff, row)
            predict_train.append(outputs.index(max(outputs)))
            #print(network)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error_train += -np.dot(expected, np.log(outputs))    # Cross entropy error
            # print(row)
            # print("expected row{}\n".format(expected))
            backward_propagate_error(network,  coeff, expected)
            update_weights(network, coeff,  row, l_rate)
        for row in test:
            outputs = forward_propagate(network, coeff, row)
            predict_test.append(outputs.index(max(outputs)))
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error_test += -np.dot(expected, np.log(outputs))    # Cross entropy error
        
        metric_values['train_accuracy'].append(accuracy_score(actual_train,predict_train ))
        metric_values['test_accuracy'].append(accuracy_score(actual_test,predict_test ))
        metric_values['train_loss'].append(sum_error_train)
        metric_values['test_loss'].append(sum_error_test)
        metric_values['param'].append(coeff['values'].copy())
        # print(" Epoch = %d  l_rate = %0.8f  error = %.2f " %(epoch, l_rate, sum_error_train) )
        # print("Coeff values" , coeff['values'])
        
    return metric_values

def back_propagation(train, test, coeff, l_rate, n_epoch, n_hidden):
    """"Backpropagation Algorithm With Stochastic Gradient Descent"""
    n_inputs = len(train.iloc[0,:])-1
    n_outputs = len(set([row[-1] for row in train.values]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    metric_values = train_network(network, coeff, train.values, test.values,  l_rate, n_epoch, n_outputs)
    # print(metric_values)
    x = [i+1 for i in range(n_epoch)]
    
    plt.plot(x, metric_values['train_accuracy'], 'go--', linewidth = 2, markersize = 6)
    # plt.plot(x, metric_values['train_loss'], 'ro-', linewidth = 2, markersize = 6)
    predictions = list()
    for row in test.values:
        prediction = predict(network,coeff, row)
        predictions.append(prediction)
    return(predictions, metric_values)

def predict(network,  coeff,  row):
        outputs = forward_propagate(network, coeff,  row)
        return outputs.index(max(outputs))

def plot_metrics(metric_values, n_epoch, cf_matrix):
    x = [i+1 for i in range(n_epoch)]
    k1 = [values[0] for values in metric_values['param']]
    k2 = [values[1] for values in metric_values['param']]
    plt.figure(1)
    plt.plot(x, metric_values['train_accuracy'], 'go--', linewidth = 2, markersize = 4, label = 'Train Accuracy')
    plt.plot(x, metric_values['test_accuracy'], 'b^--', linewidth = 2, markersize = 4, label = 'Test Accuracy')
    plt.legend()
    plt.gca().update(dict(xlabel='Epochs', ylabel='Accuracy'))
    plt.show()
    plt.figure(2)
    plt.plot(x, metric_values['train_loss'], 'go--', linewidth = 2, markersize = 4, label = 'Train loss')
    plt.plot(x, metric_values['test_loss'], 'b^--', linewidth = 2, markersize = 4, label = 'Test loss')
    plt.legend()
    plt.gca().update(dict(xlabel='Epochs', ylabel='Cross Entropy Loss'))
    plt.show()
    plt.figure(3)
    plt.plot(metric_values['train_loss'], metric_values['test_loss'], 'bo--', linewidth = 2, markersize = 4)
    plt.gca().update(dict(xlabel='Train Loss', ylabel='Test Loss'))
    plt.show()
    plt.figure(4)
    plt.plot(x, k1, 'go--', linewidth = 2, markersize = 4, label = 'K1')
    plt.plot(x, k2, 'b^--', linewidth = 2, markersize = 4, label = 'K2')
    plt.legend()
    plt.gca().update(dict(xlabel='Epochs', ylabel='Activation Parameters'))
    plt.show()
    plt.figure(5)
    heatmap(cf_matrix, annot=True)
    plt.show()
    
    

seed(2)
coeff = initialize_activationFn(2)
print("For Data: ", data_name)
print("Initial values of activation parameters k1, k2: ", coeff['values'])
l_rate = 0.1
n_epoch = 1000
n_inputs = len(train_set.iloc[0,:])-1
n_outputs = len(set([row[-1] for row in train_set.values]))
n_hidden = n_outputs + 1
print("Run parameters are : l_rate = {},  n_epoch = {}, n_inputs = {}, n_hidden = {}, n_outputs = {}".format(l_rate, n_epoch, n_inputs, n_hidden, n_outputs))
print("Balanced/unbalanced classes :", dataset.iloc[:,-1].value_counts())
predictions, metric_values = back_propagation(train_set, test_set, coeff, l_rate, n_epoch, n_hidden)

print("Final Train Accuracy is: ", metric_values['train_accuracy'][-1])
print("Final Test Accuracy is: ", accuracy_score(test_set.iloc[:,-1], predictions))
print("Final Train Loss is: ", metric_values['train_loss'][-1])
print("Final Test Loss is: ", metric_values['test_loss'][-1])
print("Final values of activation parameters k1, k2: ", metric_values['param'][-1])
cf_matrix = confusion_matrix(test_set.iloc[:,-1], predictions)
print(cf_matrix)
print(classification_report(test_set.iloc[:,-1], predictions))
plot_metrics(metric_values, n_epoch, cf_matrix)