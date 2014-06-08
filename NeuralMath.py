# William Schifeling and Daniel Alabi
import re
import nltk
import sys
import glob
import math
import cPickle as pickle
import random
import os
from NeuralNetworkClassifier import *
random.seed(0)


# function given neural network and examples, predicts for each example,
# and will tell you the average error, used for math functions.
def test_on_new_examples(network, new_examples):
    total = len(new_examples)
    error = 0.0
    for example in new_examples:
        answer = network.predict(example[0])
	error += math.fabs(1-math.fabs((answer/example[1])))
	print "Prediction = ",str(answer) +";", "Target = ", example[1]
    print "Average prediction accuracy = %.2f%%" %((1-error/total) *100)
    return error / total

# returns random x's and y's
def make_x_y(start_x, range_x, start_y, range_y):
     x = random.randint(start_x ,range_x)/200.0
     y = random.randint(start_y ,range_y)/200.0
     return (x, y)


    
def main():
    dummy_runs = ["--dummy_or", "--dummy_and", "--dummy_xor", 
                  "--dummy_nand", "--dummy_nxor", "--dummy_nor", "--dummy_add", 
                  "--dummy_subtract",  "--dummy_divide", "--dummy_multiply"]
    if len(sys.argv) < 2:
        print "python NeuralMath.py --dummy_func"
        exit()
    
    examples = []
    new_examples=[]
    
    # create examples
    if sys.argv[1] in dummy_runs:
        if sys.argv[1] == "--dummy_xor":
            examples = new_examples = [((1, 0, 0), 0),
                                       ((1, 0, 1), 1),
                                       ((1, 1, 0), 1),
                                       ((1, 1, 1), 0)]
        if sys.argv[1] == "--dummy_and":
            examples = new_examples = [((1, 0, 0), 0),
                                       ((1, 0, 1), 0),
                                       ((1, 1, 0), 0),
                                       ((1, 1, 1), 1)]        
        if sys.argv[1] == "--dummy_or":
            examples = new_examples = [((1, 0, 0), 0),
                                       ((1, 0, 1), 1),
                                       ((1, 1, 0), 1),
                                       ((1, 1, 1), 1)]

        if sys.argv[1] == "--dummy_nand":
            examples = new_examples = [((1, 0, 0), 1),
                                       ((1, 0, 1), 1),
                                       ((1, 1, 0), 1),
                                       ((1, 1, 1), 0)]  

        if sys.argv[1] == "--dummy_nxor":
            examples = new_examples = [((1, 0, 0), 1),
                                       ((1, 0, 1), 0),
                                       ((1, 1, 0), 0),
                                       ((1, 1, 1), 1)] 
	
	if sys.argv[1] == "--dummy_nor":
            examples = new_examples = [((1, 0, 0), 1),
                                       ((1, 0, 1), 0),
                                       ((1, 1, 0), 0),
                                       ((1, 1, 1), 0)]

        if sys.argv[1] == "--dummy_add":
            for i in range(100):
                (x, y) = make_x_y(0, 100, 0, 100)
                (a, b) = make_x_y(0, 100, 0, 100)
                c = a + b
                z = x + y
                examples.append(((1, x, y), z))
                new_examples.append(((1, x, y), z))
                

        if sys.argv[1] == "--dummy_subtract":
            for i in range(100):
                (x, y) = make_x_y(100, 200, 0, 100)
                (a, b) = make_x_y(100, 200, 0, 100)
                z = x - y
                c = a - b
                examples.append(((1, x, y), z))
                new_examples.append(((1, x, y), z))
                
        if sys.argv[1] == "--dummy_divide":
            for i in range(100):
                (x, y) = make_x_y(1, 100, 100, 200)
                (a, b) = make_x_y(1, 100, 100, 200)
                c = a / b
                z = x / y
                examples.append(((1, x, y), z))
                new_examples.append(((1, x, y), z))

        if sys.argv[1] == "--dummy_multiply":
            for i in range(100):
                (x, y) = make_x_y(20, 200, 50, 200)
                (a, b) = make_x_y(20, 200, 50, 200)
                c = a * b
                z = y * x
                examples.append(((1, x, y), z))
                new_examples.append(((1, x, y), z))

        

        net = NeuralNetwork([5, 5, 2, 7], examples)
        net.train()

	# test logic operators on self.
	if sys.argv[1] in dummy_runs[:6]:
            acc = net.test_on_self()
	    print "Accuracy on the logic operator", str(sys.argv[1])[8:], "was: %.2f%%" %((acc) *100) 
	    
	# test math functions on new examples.
        else:        
            acc = test_on_new_examples(net, new_examples)
    
if __name__ == "__main__":
    main()
