# William Schifeling and Daniel Alabi
import re
import nltk
import sys
import glob


'''
neuron class: contains weights, descendents (children), parents
(what it descended from), and whether or not it is an output neuron.
'''
class Neuron:

    def __init__(self):
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

    def set_input(self, boolean):
        self.is_input = boolean
    
# 1 if sum wi*ai >= 0, else 0
def hw(weights, activations):
    summation = 0.0
    for i in range(len(weights)):
        summation += weights[i]*activations[i]
    if summation >= 0:
        return 1
    else:
        return 0

# wi = wi + alpha(y-hw(w, a))*ai
def update_weight(weight_index, alpha, weights, activations, prediction):
    new_weight = weights[weight_index] + (alpha*(prediction - hw(weights, activations))*activations[weight_index])
    return new_weight





    

class NeuralNetwork:
    def __init__(self, dimensions, training_dir):
        self.dimensions = dimensions
        self.training_dir = training_dir
        self.examples = self.create_examples()
        self.network = self.create_network()
        print self.examples
        print self.network
        self.back_prop_learning()
        

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
    
        self.network = network

    def get_all_lines(self, file_name):
        f = open(file_name)
        # read subject line without 'Subject' line
        subject_line = f.readline()[8:]
        all_lines = [subject_line] \
                    + [re.sub(r"[,.]",
                              r"",
                              line.lower().strip()) for line in f]
        f.close()
        return all_lines

    def create_examples(self):
        example_list = []
        for file_name in glob.glob("/".join([self.training_dir, "*.txt"])):
            print file_name
            is_spam = 0 if file_name.find("spmsg") == -1 else 1
            all_lines = self.get_all_lines(file_name)  
            features = self.get_features(all_lines)
            example_list.append((features, is_spam))
        return example_list
    
    def get_features(self, all_lines):
        func_features = [self.get_avg_word_length, 
                         self.num_spec_chars,
                         self.num_urls,
                         self.diversity_of_chars,
                         self.num_nums,
                         self.num_dollars,
                         self.num_email_addrs,
                         self.num_space_chars]
        # for a_0
        features = [1]
        for func in func_features:
            features.append(func(all_lines))

        return tuple(features)
   


    def initialize_weights(self, weight, num_weights):
        for sub_network in self.network:
            for neuron in sub_network:
                neuron.set_weights([weight for i in range(num_weights)])
        print self.network[0][0].weights
        
                
        
    '''
    returns a network.
    '''
    def back_prop_learning(self):
        alpha = 1.0
        num_weight_is = len(self.examples[0][0])
        self.initialize_weights(0.0, num_weight_is)
        

    def get_avg_word_length(self, all_lines):
        num_words = 0.0
        length_sum = 0.0
        for line in all_lines:
            no_punc = re.sub(r"\W", r" ", line)
            for word in nltk.word_tokenize(no_punc):
                num_words += 1
                length_sum += len(word)
        return length_sum/num_words


    def num_spec_chars(self, all_lines):
        num_chars = 0.0
        for line in all_lines:
            spec_chars = re.sub(r"\w", r" ", line)
            for char in nltk.word_tokenize(spec_chars):
                num_chars += 1
        return num_chars

    def num_urls(self, all_lines):
        num_urls = 0.0
        for line in all_lines:
            for word in nltk.word_tokenize(line):
                if (word == "http") or (word == "www"): #.com
                    num_urls += 1
        return num_urls
            
        
    def diversity_of_chars(self, all_lines):
        chars = set()
        num_chars = 0.0
        for line in all_lines:
            for word in nltk.word_tokenize(line):
                for char in word:
                    chars.add(ord(char))
                    num_chars +=1
        return len(chars)/num_chars

    def num_nums(self, all_lines):
        num_nums = 0.0
        for line in all_lines:
            num_list = re.findall("[0-9]", line)
            num_nums += len(num_list)
        return num_nums

    def num_dollars(self, all_lines):
        num_dollars = 0.0
        for line in all_lines:
            dollar_list = re.findall("$", line)
            num_dollars += len(dollar_list)
        return num_dollars

    def num_email_addrs(self, all_lines):
        num_emails = 0.0
        for line in all_lines:
            email_list = re.findall("@", line)
            num_emails += len(email_list)
        return num_emails

    def num_space_chars(self, all_lines):
        num_spaces = 0.0
        for line in all_lines:
            space_list = re.findall("[\t, \s, \n]", line)
            num_spaces += len(space_list)
        return num_spaces
            

    
    
    
def main():
    if len(sys.argv) < 3:
        print "python [Classifier File Name] [training dirs] [testing dir]"
        exit()

    training_dir = sys.argv[1]
    testing_dir = sys.argv[2]
    
    network = NeuralNetwork([4, 2, 2], training_dir)
'''    acc = network.test_classifier()
    print "Accuracy on test set: %.2f %%" % (acc * 100)
    
    network = createNeuralNetwork(3, 3)
    print network
'''
main()
