# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        Cweights = {}
        Cscores = {}
        
        def calcTau(cee, wya, wy, f):
            return min(cee, (((wya - wy) * f + 1.0) / (2 * (f * f))))
        
        for c in Cgrid:
            Cweights[c] = self.weights.copy()
            Cscores[c] = 0
            for i in range(self.max_iterations):
                print "Iteration ", i, " ", c
                for j in range(len(trainingData)):
                    results = util.Counter()
                    
                    for l in self.legalLabels:
                        results[l] = (trainingData[j] * Cweights[c][l], l)
                    
                    est = results.argMax()
                    
                    if(trainingLabels[j] != est):
                        #print j
                        tau = calcTau(c, Cweights[c][est], Cweights[c][trainingLabels[j]], trainingData[j])
                        
                        temp = trainingData[j]
                        for t in temp:
                            temp[t] *= tau
                            
                        Cweights[c][trainingLabels[j]] = Cweights[c][trainingLabels[j]] + temp
                        Cweights[c][est] = Cweights[c][est] - temp
            
        
        for c in Cgrid:
            self.weights = Cweights[c].copy()
            valResults = self.classify(validationData)
            for i in range(len(validationData)):
                if(valResults[i] == validationLabels[i]):
                    Cscores[c] += 1
        
        bestScore = 0
        bestC = Cgrid[0]
        for c in Cgrid:
            print Cscores[c]
            if(Cscores[c] > bestScore):
                bestScore = Cscores[c]
                bestC = c
        
        #print "Best C is ", bestC
        self.weights = Cweights[bestC].copy()

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []
        
        weights = []
        
        for w in self.weights[label]:
            weights.append((w,self.weights[label][w]))
        
        #SHOULD HAVE USED COUNTER.SORTEDKEYS D:
        #sorts by the second value of a tuple, switch to desc order
        sortedWeights = sorted(weights, key=lambda ((x,y),z): z, reverse=True)
        
        for i in range(100):
            featuresWeights.append(sortedWeights[i][0])
        
        return featuresWeights        

