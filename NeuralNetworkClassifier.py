'''
Daniel Alabi and Will Schifeling 
CS 321

NeuralNetworkClassifier.py -->
Implementation of a Neural Network Classifier

Example Run:
(without cross validation)
python NeuralNetworkClassifier.py ./spambase/smallestspambase.data 
-------------------OR-------------------
(with cross validation -- do on really small files)
python NeuralNetworkClassifier.py ./spambase/smallestspambase.data --cross
'''

import re
import nltk
import sys
import glob
import math
import cPickle as pickle
import random
import os

# seed the random number generator
random.seed(0)

class Node:
    '''
    Represents one node in a Neural Network
    '''
    def __init__(self):
        # weights going out from this node
        self.weights = []
        self.is_output = False
        self.is_input = False

    def set_weights(self, weightList):
        self.weights = weightList

    def set_output(self, boolean):
        self.is_output = boolean
        self.weights = None

    def set_input(self, boolean):
        self.is_input = boolean

    

class NeuralNetwork:
    '''
    Represents a Multi-Layer Neural Network that uses the
    back-propagation algorithm to train the Neural Netwok.
    You can have as many hidden layers as you want.
    ----------------------------------------------------------------
    We add in an input layer with as many nodes as there are inputs.    
    We add in an output layer with only one node for the output
    classification.
    ----------------------------------------------------------------
    '''
    def __init__(self, dimensions, examples):
        self.examples = examples

        # add input layer to dimensions
        self.dimensions = [len(self.examples[0][0])]
        self.dimensions += dimensions

        # create network based on input and hidden layers
        # adds in an extra output layer
        self.network = self.create_network()

        # add output layer to dimensions
        self.dimensions += [1]

        # True if you want to see the weight changes. False otherwise
        self.verbose = True
                
    def create_network(self):
        '''
        Creates the network of nodes based on the dimensions
        the user specifies.
        '''
        network = []
        for number in self.dimensions:
            sub_network = []
            for i in range(number):
                neuron = Node()
                sub_network.append(neuron)
            network.append(sub_network)
                
        # create output node for the output layer
        neuron = Node()
        neuron.set_output(True)
        network.append([neuron])

        # set the input layer 
        for neuron in network[0]:
            neuron.set_input(True)
    
        # connect the layers together by weights
        for i in range(len(network)):
            for neuron in network[i]:
                if i < len(network)-1:
                    # initialize weights to some random number
                    # between -0.5 and 0.5
                    weights = []
                    for j in range(len(network[i+1])):
                        random_num = random.random() - 0.5  
                        weights.append(random_num)
                    neuron.set_weights(weights)
    
        return network
                
    def sigmoid(self, x):
        '''
        sigmoid function -> Continuous Tan-Sigmoid Function
        see 
        http://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions
        '''
        return math.tanh(x)
  
    def dsigmoid(self, y):
        '''
        derivative of sigmoid function on the output
        '''
        return 1.0 - y**2
        
    
    def forward_pass(self, input_vec):
        '''
        Perform a forward pass of the neural network using the 'input_vec'
        '''
        ## set activations for input, hidden, and output nodes

        # input nodes
        a = [[None] * len(self.network[l]) for l in range(len(self.network))]
        for j in range(len(input_vec)):
            a[0][j] = input_vec[j]
                
        # hidden and output nodes
        for l in range(1, len(self.dimensions)): 
            prev_layer = self.network[l-1]
            curr_layer = self.network[l]                
            
            for j in range(len(curr_layer)):
                # compute the hypothesis for this layer
                hyp = 0
                for i in range(len(prev_layer)):
                    hyp += prev_layer[i].weights[j] * a[l-1][i]            
                a[l][j] = self.sigmoid(hyp)
                
        return a

    def back_propagate(self, a, res):
        '''
        Compute the error on the output node and then back propagate this
        error throughout the network.
        We essentially used the book's back propagation algorithm.
        '''
        # learning rate
        alpha = 0.3    
        # delta -> store the errors 
        delta = [[None] * len(self.network[l]) for l in range(len(self.network))]

        # keep track of the weight change
        weight_change = 0

        output_node = self.network[-1][0]
        # error on output node
        error = res - a[-1][0]        
        delta[-1][0] = error * self.dsigmoid(a[-1][0])
        
        # from layer L-2 to layer 0, propagate the error
        for l in range(len(self.network)-2, -1, -1):
            for j in range(len(self.network[l])):
                error = 0
                for k in range(len(self.network[l+1])):
                    node = self.network[l][j]
                    error += node.weights[k] * delta[l+1][k]
                delta[l][j] = self.dsigmoid(a[l][j]) * error     

        # update the weights based on the delta errors
        for l in range(len(self.network)-1):
            for j in range(len(self.network[l])):
                node = self.network[l][j]
                for k in range(len(node.weights)):                      
                    wc = alpha * a[l][j] * delta[l+1][k]
                    node.weights[k] += wc
                    weight_change += math.fabs(wc)

        return weight_change

    def get_model_file_name(self, training_file):
        '''
        Return a file name to store the current model
        '''
        ret = training_file
        for num in self.dimensions:
            ret += "-" + str(num)
        ret += ".dat"
        return ret

    def model_exists(self, training_file):
        '''
        Return True
        if a model should exist for this Neural Network
        '''
        file_name = self.get_model_file_name(training_file)
        return os.path.exists(file_name)        

    def store_model(self, training_file):
        '''
        Store this model  
        '''
        file_name = self.get_model_file_name(training_file)
        if os.path.exists(file_name):
            os.remove(file_name)
        
        # store all the weights for each node in the network
        network_weights = []
        for layer in self.network:
            layer_weights = []
            for node in layer:
                layer_weights.append(node.weights)
            network_weights.append(layer_weights)

        pickle.dump(network_weights, 
                    open(file_name, "wb")
        )
        
        
    def restore_model(self, training_file):
        '''
        Restore the model made for this file
        '''
        file_name = self.get_model_file_name(training_file)
        network_weights = pickle.load(
            open(file_name, "rb")
        )
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                node = self.network[i][j]
                node.weights = network_weights[i][j]

    def train(self):                
        '''
        Train this model using the examples
        '''
        # initialize weight change
        weight_change = 1
        
        # keep track of number of iterations 
        times = 0           
        MAX_ITER = 1000
        while math.fabs(weight_change) > 0 and times < MAX_ITER:
            for example in self.examples:
                input_vec = example[0]            
                res = example[1]

                # forward pass
                a = self.forward_pass(input_vec)
                
                # propagate backwards...
                weight_change = self.back_propagate(a, res)                

            if self.verbose and times % (MAX_ITER/20) == 0:
                print "Weight Change -->", weight_change
            times += 1
                            
    def predict(self, features):
        '''
        Predict the result of the network based on the
        features
        '''
        forwards = self.forward_pass(features)
        return forwards[-1][0] 

    def test_on(self, examples):
        '''
        Test on the examples and return the accuracy
        '''
        total = len(examples)
        right = 0.0
        for example in examples:
            answer = self.predict(example[0])
            round_answer = 1 if answer >= 0.5 else 0
            if round_answer == example[1]:
                right += 1                
        return right / total
        
    def test_on_self(self):        
        '''
        Test on itself. A way to make sure our Neural Network 
        is doing good.
        '''
        print "Testing on training set..."
        return self.test_on(self.examples)

    def random_sep(self, rat = .9):
        '''
        separate the examples into
        training and testing set
        90% -> training
        10% -> testing
        by default
        '''
        print "Seperating examples into %.2f%%:%.2f%% (Training/Testing)" % \
            (rat * 100, (1-rat)*100)
        training = []
        chosen_indices = set()
        max_indices = int(rat * len(self.examples))

        while len(chosen_indices) < max_indices:
            sel_index = random.randint(0, len(self.examples)-1)
            if sel_index not in chosen_indices:
                chosen_indices.add(sel_index)
                training.append(self.examples[sel_index])

        testing = []
        for example in self.examples:
            if example not in training:
                testing.append(example)

        return training, testing

    def test_cross_validation(self):
        '''
        Cross validation for 5 iterations
        '''
        print "\nTesting using cross-validation...\n"
        accs = []

        for i in range(5):
            print "iteration %d" % (i + 1)
            train_examples, test_examples = self.random_sep()
            net = NeuralNetwork([10, 10], train_examples)
            net.train()
            acc = net.test_on(test_examples)
            accs.append(acc)
        return sum(accs) / float(len(accs))


def normalize_features(example_list):
    '''
    Normalize all columns in example_list 
    '''
    max_list = [[0.0] * len(example_list[0][0]), 0.0]
    for row in range(len(example_list)):
        for col in range(len(example_list[row][0])):
            cur = math.fabs(example_list[row][0][col])
            if cur > max_list[0][col]:
                max_list[0][col] = cur
        cur = math.fabs(example_list[row][1])
        if cur > max_list[1]:
            max_list[1] = cur
    
    for row in range(len(example_list)):
        for col in range(len(example_list[row][0])):
            if max_list[0][col] != 0:
                example_list[row][0][col] = example_list[row][0][col] / max_list[0][col]
        example_list[row][1] =  example_list[row][1] / max_list[1]
    
def create_examples(training_file):
    '''
    Create the examples based on the 'training_file'
    '''
    example_list = []
    with open(training_file) as f:
        for line in f:
            row = line.split(",")
            features = [1.0]+[float(each) for each in row[:-1]]
            is_spam = int(row[-1])
            example_list.append([features, is_spam])

    # normalize the features
    normalize_features(example_list)
    
    return example_list

    
def main():
    if len(sys.argv) < 2:
        print "python NeuralNetworkClassifier.py [training file]"
        exit()

    # training file
    training_file = sys.argv[1]
    # get examples based on the training file
    examples = create_examples(training_file)

    # create the network with 2 hidden layers, each with 10 nodes
    net = NeuralNetwork([10, 10], examples)
    if not net.model_exists(training_file):
        net.train()
        net.store_model(training_file)
    else:
        net.restore_model(training_file)
    # test on self
    acc = net.test_on_self()
    print "Accuracy on training data: %d%%" % (acc * 100)

    if len(sys.argv) >= 3 and sys.argv[2] == "--cross":
        # cross-validation
        acc = net.test_cross_validation()
        print "Accuracy using cross-validation: %d%%" % (acc * 100)
    

if __name__ == "__main__":
    main()
