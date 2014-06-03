'''
Daniel Alabi and Will Schifeling
Inspired by sentimentstwitter on github.com/alabid
'''
import sys
import nltk
import os
import cPickle as pickle
import random
import glob
import re
import math

class NaiveBayesClassifier:
    '''
    rawdname -> name of directory containing raw training data
    force == True iff user wants to overwrite classifier data
    '''
    def __init__(self, dname, *args, **kargs):
        self.rawdname = dname

        self.force = kargs.get("force", False)

        # name of (pickled) file containing model
        self.modelfname = "model.dat"
        self.weight = kargs.get("weight", 0.00005)

        # to avoid classifying too many 'documents' as
        # spam or not spam, we use a threshold 
        self.thresholds = [1.0, 1.0]

        # message counts in each category
        self.mess_counts = [0, 0]
        
        # counts of feature/class combinations
        # stores (feature) => [x, y] where
        # x -> number of times feature appears in negative class
        # y -> number of times feature appears in positive class
        self.feat_mess_counts = {}

    def setThresholds(self, neg=1.0, pos=1.0):
        self.thresholds = [neg, pos]

    def probMessageClass(self, text, c):
        '''
        Returns the (log) probability of a message, given
        a particular class
        P(message | class)
        '''
        features = self.getFeatures(text)
        p = 0
        for f in features:
            p += math.log(self.weightedProb(f, c))
        return p

    def probClassMessage(self, text, c):
        '''
        Returns the (log) probability of a class, given 
        a particular message
        P(class | message) = P(message | class) X P(class) / P(message)
        But P(message) is constant for all classes; so forget
        '''
        return self.probMessageClass(text, c) + math.log(self.probC(c))

    def classify(self, text):
        '''
        Returns 0 (negative) if
           P(class=0 | message) > P(class=1 | message) * thresholds[0]
        Returns 1 (positive) if
           P(class=1 | message) > P(class=0 | message) * thresholds[1]
        Else return -1 (neutral)
        '''
        p0 = self.probClassMessage(text, 0)
        p1 = self.probClassMessage(text, 1)
        
        if p0 > p1 + math.log(self.thresholds[0]):
            return 0
        elif p1 > p0 + math.log(self.thresholds[1]):
            return 1
        else:
            return -1

    def incFC(self, f, c):
        self.feat_mess_counts.setdefault(f, [0, 0])
        self.feat_mess_counts[f][c] += 1

    def incC(self, c):
        self.mess_counts[c] += 1

    def getFC(self, f, c):
        if f in self.feat_mess_counts:
            return float(self.feat_mess_counts[f][c])
        return 0.0

    def getC(self, c):
        return float(self.mess_counts[c])

    def getTotal(self):
        return sum(self.mess_counts)

    def getFeatures(self, item):
        '''
        Each feature has weight 1.
        That is, even if the word 'obama' appears > 10 times
        in a message, it is counted only once in that particular tweet
        '''
        flist = []
        return nltk.word_tokenize(item)

    def train(self, c, item):
        '''
        Trains the classifier using item (a line) on the class 'c'
        '''
        features = self.getFeatures(item)
        for f in features:
            self.incFC(f, c)
        self.incC(c)

    def testClassifier(self, fname):
        total = 0
        right = 0
        for file_name in glob.glob("/".join([fname, "*.txt"])):
            is_spam = 0 if file_name.find("spmsg") == -1 else 1
            all_lines = self.get_all_lines(file_name)            

            for line in all_lines:
                if self.classify(line) == is_spam:
                    right += 1
                total += 1
        print "Testing model on the following directories: "
        for dirname in glob.glob(fname):
            print dirname
        return float(right)/total

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

    def trainClassifier(self):
        '''
        Trains the classifier based on messages in the directory.
        Stores the resulting data structure in a pickle file.
        '''
        if self.force:
            if os.path.exists(self.modelfname):
                os.remove(self.modelfname)
        elif os.path.exists(self.modelfname):
            self.mess_counts, self.feat_mess_counts = pickle.load(
                open(self.modelfname, "rb")
            )
            return
        
        for file_name in glob.glob("/".join([self.rawdname, "*.txt"])):   
            is_spam = 0 if file_name.find("spmsg") == -1 else 1
            all_lines = self.get_all_lines(file_name)            

            for line in all_lines:
                self.train(is_spam, line)

        # store naive bayes classifier training data
        pickle.dump([self.mess_counts, self.feat_mess_counts],
                    open(self.modelfname, "wb")
        )
        print "Creating model based on the following directories: "
        for dirname in glob.glob(self.rawdname):
            print dirname
        print "Model stored in '%s'" % self.modelfname

    def probFC(self, f, c):
        '''
        Return the probability of a feature being in a particular class
        '''
        if self.getC(c) == 0:
            return 0
        return self.getFC(f, c) / self.getC(c)

    def probC(self, c):
        '''
        Return the probability Prob(Class)
        '''
        return self.getC(c) / self.getTotal()

    def setWeight(self, w):
        '''
        Set weight to use in classifier
        '''
        self.weight = w

    def weightedProb(self, f, c, ap=0.5):
        '''
        Method of smoothing:
        Start with an assumed probability (ap) for each word in each class
        Then, return weighted probability of real probability (probFC)
        and assumed probability
        weight of 1.0 means ap is weighted as much as a word
        Bayesian in nature: 
        For example, the word 'dude' might not be in the corpus initially.
        so assuming weight of 1.0, then
        P('dude' | class=0) = 0.5 and P('dude' | class=1) = 0.5
        then when we find one 'dude' that's positive,
        P('dude' | class=0) = 0.25 and P('dude' | class=1) = 0.75
        '''
        # calculate current probability
        real = self.probFC(f, c)

        # count number of times this feature has appeared in all categories
        totals = sum([self.getFC(f, c) for c in [0, 1]])
        
        # calculate weighted average
        return ((self.weight * ap) + (totals * real))/(self.weight + totals)

    def __repr__(self):
        return "Classifier Info: (weight=%s, thresholds=%s)" % (self.weight,
                                                                self.thresholds)


def main():
    if len(sys.argv) < 3:
        print "python [Classifier File Name] [training dirs] [testing dir]"
        exit()
    naive = NaiveBayesClassifier(sys.argv[1])
    naive.force = True
    naive.trainClassifier()
    acc = naive.testClassifier(sys.argv[2])
    print "Accuracy on test set: %.2f %%" % (acc * 100)

if __name__ == "__main__":
    main()
