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



def test_on_new_examples(network, new_examples):
    total = len(new_examples)
    error = 0.0
    for example in new_examples:
        answer = network.predict(example[0])
	error += math.fabs(1-(answer/example[1]))
	print "Prediction = ",str(answer) +";", "Target = ", example[1]
    print "Average prediction accuracy = %.2f%%" %((1-error/total) *100)
    return error / total

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
                (x, y) = make_x_y(50, 100, 0, 50)
                (a, b) = make_x_y(50, 100, 0, 50)
                z = x - y
                c = a - b
                examples.append(((1, x, y), z))
                new_examples.append(((1, x, y), z))
                
        if sys.argv[1] == "--dummy_divide":
            for i in range(100):
                (x, y) = make_x_y(1, 50, 50, 100)
                (a, b) = make_x_y(1, 50, 50, 100)
                c = a / b
                z = x / y
                examples.append(((1, x, y), z))
                new_examples.append(((1, x, y), z))

        if sys.argv[1] == "--dummy_multiply":
            for i in range(100):
                (x, y) = make_x_y(1, 100, 1, 100)
                (a, b) = make_x_y(1, 100, 1, 100)
                c = a * b
                z = y * x
                examples.append(((1, x, y), z))
                new_examples.append(((1, x, y), z))

        

        net = NeuralNetwork([5, 5, 2, 7], examples)
        net.train()
	if sys.argv[1] in dummy_runs[:6]:
            acc = net.test_on_self()
	    print "Accuracy on the logic operator was: %.2f%%" %((acc) *100) 
        else:        
            acc = test_on_new_examples(net, new_examples)
    
if __name__ == "__main__":
    main()
