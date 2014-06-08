# William Schifeling and Daniel Alabi
import re
import nltk
import sys
import glob
import math
import cPickle as pickle
import random
import os

random.seed(0)

'''
neuron class: contains weights, descendents (children), parents
(what it descended from), and whether or not it is an output neuron.
'''
class Neuron:

    def __init__(self):
        # weights going out from this Neuron
        self.weights = []
        self.children = []
        self.parents = []
        self.is_output = False
        self.is_input = False

    def set_weights(self, weightList):
        self.weights = weightList

    def set_children(self, children):
        self.children = children

    def set_parents(self, parents):
        self.parents = parents

    def set_output(self, boolean):
        self.is_output = boolean
        self.weights = None

    def set_input(self, boolean):
        self.is_input = boolean

    

class NeuralNetwork:
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
        
    def create_network(self):
        network = []
        for number in self.dimensions:
            sub_network = []
            for i in range(number):
                neuron = Neuron()
                sub_network.append(neuron)
            network.append(sub_network)
                
        neuron = Neuron()
        neuron.set_output(True)
        network.append([neuron])

        for neuron in network[0]:
            neuron.set_input(True)
    
        for i in range(len(network)):
            for neuron in network[i]:
                if i > 0:
                    neuron.set_parents(network[i-1])                    
                if i < len(network)-1:
                    neuron.set_children(network[i+1])
                    # initialize weights to some random number
                    # between -0.5 and 0.5
                    weights = []
                    for j in range(len(network[i+1])):
                        random_num = random.random() - 0.5  
                        weights.append(random_num)
                    neuron.set_weights(weights)
    
        return network
                
    def sigmoid(self, x):
        return math.tanh(x)
  
    def dsigmoid(self, y):
        return 1.0 - y**2
        
    def forward_pass(self, input_vec):
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
                hyp = 0
                for i in range(len(prev_layer)):
                    hyp += prev_layer[i].weights[j] * a[l-1][i]
       
                a[l][j] = self.sigmoid(hyp)
        return a

    def back_propagate(self, a, res):
        alpha = 0.3
        delta = [[None] * len(self.network[l]) for l in range(len(self.network))]

        weight_change = 0
        output_node = self.network[-1][0]
        error = res - a[-1][0]
        delta[-1][0] = error * self.dsigmoid(a[-1][0])
        
        for l in range(len(self.network)-2, -1, -1):
            for j in range(len(self.network[l])):
                error = 0
                for k in range(len(self.network[l+1])):
                    node = self.network[l][j]
                    error += node.weights[k] * delta[l+1][k]
                delta[l][j] = self.dsigmoid(a[l][j]) * error     

        for l in range(len(self.network)-1):
            for j in range(len(self.network[l])):
                node = self.network[l][j]
                for k in range(len(node.weights)):                      
                    wc = alpha * a[l][j] * delta[l+1][k]
                    node.weights[k] += wc
                    weight_change += math.fabs(wc)

        return weight_change

    def get_model_file_name(self, training_file):
        ret = training_file
        for num in self.dimensions:
            ret += "-" + str(num)
        ret += ".dat"
        return ret

    # Return True
    # if a model should exist for this Neural Network
    def model_exists(self, training_file):
        file_name = self.get_model_file_name(training_file)
        return os.path.exists(file_name)        

    def store_model(self, training_file):
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
        file_name = self.get_model_file_name(training_file)
        network_weights = pickle.load(
            open(file_name, "rb")
        )
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                node = self.network[i][j]
                node.weights = network_weights[i][j]

    def train(self):                
        # keep track of difference in weights
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

            if times % (MAX_ITER/20) == 0:
                print "Weight Change -->", weight_change
            times += 1
                            
    def predict(self, features):
        forwards = self.forward_pass(features)
        return forwards[-1][0] 

    def classify(self, all_lines):
        features = get_features(all_lines)
        output = self.predict(features)
        return output

    def test_on(self, fname):
        total = 0
        right = 0
        for file_name in glob.glob("/".join([fname, "*.txt"])):
            is_spam = 0 if file_name.find("spmsg") == -1 else 1
            all_lines = get_all_lines(file_name)            
            
            answer = self.classify(all_lines)        
            round_answer = 1 if answer >= 0.5 else 0
            if round_answer == is_spam:
                right += 1
            total += 1
        print "Testing model on the following directories: "
        for dirname in glob.glob(fname):
            print dirname
        return float(right)/total    

    def test_on_self(self):        
        total = len(self.examples)
        right = 0.0
        for example in self.examples:
            answer = self.predict(example[0])
            round_answer = 1 if answer >= 0.5 else 0
            if round_answer == example[1]:
                right += 1
            
        return right / total

def normalize_features(example_list):
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
    example_list = []
    # for file_name in glob.glob("/".join([from_dir, "*.txt"])):
    with open(training_file) as f:
        for line in f:
            row = line.split(",")
            features = [1.0]+[float(each) for each in row[:-1]]
            is_spam = int(row[-1])
            example_list.append([features, is_spam])

    normalize_features(example_list)

    return example_list

def get_all_lines(file_name):
    f = open(file_name)
    # read subject line without 'Subject' line
    subject_line = f.readline()[8:]
    all_lines = [subject_line] \
                + [re.sub(r"[,.]",
                          r"",
                          line.lower().strip()) for line in f]
    f.close()
    return all_lines
    
    
def get_features(all_lines):        
    func_features = [get_avg_word_length,
                     num_spec_chars,
                     num_urls,
                     num_nums,
                     num_dollars,
                     num_email_addrs,
                     diversity_of_chars]

    for word in ["free", "money", "credit", "buy", "sell",
                 "sex", "purchase", "job", "call", "business",
                 "save", "cash", "cost", "quick", "easy"]:
        func_features.append(word_count_generate(word, all_lines))

    # for a_0
    features = [1]
    for func in func_features:
        features.append(func(all_lines))
        
    return features

def word_count_generate(word, all_lines):
    def word_count(all_lines):        
        num_word = 0.0
        for line in all_lines:
            if word in line:
                num_word += 1
        return num_word
    return word_count

def get_avg_word_length(all_lines):
    num_words = 0.0
    length_sum = 0.0
    for line in all_lines:
        no_punc = re.sub(r"\W", r" ", line)
        for word in nltk.word_tokenize(no_punc):
            num_words += 1
            length_sum += len(word)
    return length_sum/num_words


def num_spec_chars(all_lines):
    num_chars = 0.0
    for line in all_lines:
        spec_chars = re.sub(r"\w", r" ", line)
        for char in nltk.word_tokenize(spec_chars):
            num_chars += 1
    return num_chars

def num_urls(all_lines):
    num_urls = 0.0
    for line in all_lines:
        for word in nltk.word_tokenize(line):
            if (word == "http") or (word == "www"): #.com
                num_urls += 1
    return num_urls
                    
def diversity_of_chars(all_lines):
    chars = set()
    for line in all_lines:
        for word in nltk.word_tokenize(line):
            for char in word:
                chars.add(ord(char))
    return len(chars)
    
def num_nums(all_lines):
    num_nums = 0.0
    for line in all_lines:
        num_list = re.findall("[0-9]", line)
        num_nums += len(num_list)
    return num_nums
    
def num_dollars(all_lines):
    num_dollars = 0.0
    for line in all_lines:
        dollar_list = re.findall("$", line)
        num_dollars += len(dollar_list)
    return num_dollars
    
def num_email_addrs(all_lines):
    num_emails = 0.0
    for line in all_lines:
        email_list = re.findall("@", line)
        num_emails += len(email_list)
    return num_emails


    
def main():
    dummy_runs = ["--dummy_or", "--dummy_and", "--dummy_xor", 
                  "--dummy_nand", "--dummy_nxor", "--dummy_add"]
    if len(sys.argv) < 2:
        print "python NeuralNetworkClassifier.py [training dir]"
        print "-----------OR----------"
        print "python NeuralNetworkClassifier.py --dummy_func"
        exit()
    
    if sys.argv[1] in dummy_runs:
        if sys.argv[1] == "--dummy_xor":
            examples = [((1, 0, 0), 0),
                        ((1, 0, 1), 1),
                        ((1, 1, 0), 1),
                        ((1, 1, 1), 0)]
        if sys.argv[1] == "--dummy_and":
            examples = [((1, 0, 0), 0),
                        ((1, 0, 1), 0),
                        ((1, 1, 0), 0),
                        ((1, 1, 1), 1)]        
        if sys.argv[1] == "--dummy_or":
            examples = [((1, 0, 0), 0),
                        ((1, 0, 1), 1),
                        ((1, 1, 0), 1),
                        ((1, 1, 1), 1)]

        if sys.argv[1] == "--dummy_nand":
            examples = [((1, 0, 0), 1),
                        ((1, 0, 1), 1),
                        ((1, 1, 0), 1),
                        ((1, 1, 1), 0)]  

        if sys.argv[1] == "--dummy_nxor":
            examples = [((1, 0, 0), 1),
                        ((1, 0, 1), 0),
                        ((1, 1, 0), 0),
                        ((1, 1, 1), 1)] 

        if sys.argv[1] == "--dummy_add":
            examples = []
            for i in range(100):
                x = random.randint(0,100)/200.0
                y = random.randint(0,100)/200.0
                z = x + y
                examples.append(((1, x, y), z))        

        net = NeuralNetwork([5, 10, 5], examples)
        net.train()
        acc = net.test_on_self()

    else:

        training_file = sys.argv[1]
        examples = create_examples(training_file)

        net = NeuralNetwork([10, 10], examples)
        if not net.model_exists(training_file):
            net.train()
            net.store_model(training_file)
        else:
            net.restore_model(training_file)
        acc = net.test_on_self()

    print "Accuracy on test set: %.2f %%" % (acc * 100)
    

if __name__ == "__main__":
    main()
