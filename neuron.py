# William Schifeling and Daniel Alabi

'''
neuron class: contains weights, descendents (children), parents
(what it descended from), and whether or not it is an output neuron.
'''
class Neuron():

    def __init__(self):
        self.weights = []
        self.children = []
        self.parents = []
        self.isOutput = False
        self.isInput = False

    def setWeight(self, weightList):
        self.weights = weightList

    def setChildren(self, children):
        self.children = children

    def setParents(self, parents):
        self.parents = parents

    def setOutput(self, boolean):
        self.isOutput = boolean

    def setInput(self, boolean):
        self.isInput = boolean
    
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
def updateWeight(weightIndex, alpha, weights, activations, prediction):
    newWeight = weights[weightIndex] + (alpha*(prediction - hw(weights, activations))*activations[weightIndex])
    return newWeight

def feedForwardNetwork(examples, alpha):
    pass
    '''
    network = createNeuralNetwork(1, 1)
    inputNode = network[0][0]
    inputNode.setWeights([0 for item in examples[0]])
    iters, changes = 0, 1
    while (iters < 100) or (changes = 0):
        changes = 0
        for example in examples:
            for weight in inputNode.weights:
     '''           



'''
creates a neural network with depth equal to "depth" + 1, and width equal
to the "width". Depth = depth + 1 because we have one final output node for now
so we don't have to deal with a vector output.
'''
def createNeuralNetwork(depth, width):
    network = []
    for i in range(depth):
        subnetwork = []
        for j in range(width):
            neuron = Neuron()
            subnetwork.append(neuron)
        network.append(subnetwork)
    neuron = Neuron()
    neuron.setOutput(True)
    network.append([neuron])

    for neuron in network[0]:
        neuron.setInput(True)
    
    for i in range(len(network)):
        for neuron in network[i]:
            if i > 0:
                neuron.setParents(network[i-1])
            if i < depth:
                neuron.setChildren(network[i+1])
    
    return network
    
    
def main():
    network = createNeuralNetwork(3, 3)
    print network

main()
