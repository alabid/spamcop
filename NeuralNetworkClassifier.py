# William Schifeling and Daniel Alabi
import re
import nltk
import sys
import glob
import math
import random

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
                    random_num = random.random() - 0.5                    
                    neuron.set_weights([random_num] * len(network[i+1]))
    
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


    def train(self):
        alpha = 0.5
        delta = [[None] * len(self.network[l]) for l in range(len(self.network))]
        
        # keep track of difference in weights
        weight_change = 1
        
        # keep track of number of iterations 
        times = 0           
        MAX_ITER = 1000
        while math.fabs(weight_change) > 0 and times < MAX_ITER:
            weight_change = 0
            for example in self.examples:
                input_vec = example[0]            
                res = example[1]

                # forward pass
                a = self.forward_pass(input_vec)
                
                # propagate backwards...
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

            times += 1
            if times % (MAX_ITER/20) == 0:
                print "Weight Change -->", weight_change
                            
    def predict(self, features):
        forwards = self.forward_pass(features)
        return forwards[-1][0] 

    def classify(self, text):
        features = get_features(text)
        output = self.predict(features)
        return output

    def test_on(self, fname):
        total = 0
        right = 0
        for file_name in glob.glob("/".join([fname, "*.txt"])):
            is_spam = 0 if file_name.find("spmsg") == -1 else 1
            all_lines = get_all_lines(file_name)            
            
            all_text = "\n".join(all_lines)
            if self.classify(all_text) == is_spam:
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
            print "answer=", answer, ";target=", example[1]
            if round_answer == example[1]:
                right += 1
            
        return right / total

def create_examples(from_dir):
    example_list = []
    for file_name in glob.glob("/".join([from_dir, "*.txt"])):
        is_spam = 0 if file_name.find("spmsg") == -1 else 1
        all_lines = get_all_lines(file_name)  
        features = get_features(all_lines)
        example_list.append((features, is_spam))
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
                     diversity_of_chars,
                     num_nums,
                     num_dollars,
                     num_email_addrs,
                     num_space_chars]
    # for a_0
    features = [1]
    for func in func_features:
        features.append(func(all_lines))
        
    return tuple(features)        
        
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
    num_chars = 0.0
    for line in all_lines:
        for word in nltk.word_tokenize(line):
            for char in word:
                chars.add(ord(char))
                num_chars +=1
    return len(chars)/num_chars
    
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

def num_space_chars(all_lines):
    num_spaces = 0.0
    for line in all_lines:
        space_list = re.findall("\s+", line)
        num_spaces += len(space_list)
    return num_spaces

    
def main():
    if len(sys.argv) < 3 and not (len(sys.argv) == 2 and sys.argv[1] == "--dummy"):
        print "python NeuralNetworkClassifier.py [training dirs] [testing dir]"
        print "-----------OR----------"
        print "python NeuralNetworkClassifier.py --dummy"
        exit()
    
    if sys.argv[1] == "--dummy":
        examples = [((1, 0, 0), 0),
                    ((1, 0, 1), 1),
                    ((1, 1, 0), 1),
                    ((1, 1, 1), 0)]

        network = NeuralNetwork([2,2], examples)
        network.train()
        acc = network.test_on_self()

    else:

        training_dir = sys.argv[1]
        testing_dir = sys.argv[2]        
        examples = create_examples(training_dir)

        network = NeuralNetwork([2, 2], examples)
        network.train()        
        acc = network.test_on(testing_dir)

    print "Accuracy on test set: %.2f %%" % (acc * 100)
    

if __name__ == "__main__":
    main()
