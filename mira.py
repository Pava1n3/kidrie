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
        cweights = {}
        cscores = {}
        for c in Cgrid:
            cweights[c] = self.weights.copy()
            cscores[c] = 0
        
        def calcT(c, i):
            #print type(trainingData[i])
            return min(c, ((cweights[c][guessedLabel] - cweights[c][trainingLabels[i]]) * trainingData[i] + 1.0) / (2.0 * (trainingData[i] * trainingData[i])))
        
        def tMaalF(t, Ef):
            for f in Ef:
                Ef[f] *= t
            return Ef
        
        for c in Cgrid:
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):
                    scores = util.Counter()
                    
                    for l in self.legalLabels:
                        #determine the score for a label for trainingData[i] and add it to a list of scores
                        score = util.Counter()
                        score = trainingData[i] * cweights[c][l]
                        scores[l] = score
                    
                    #from the list of scores, get the label with the highest score
                    guessedLabel = scores.argMax()
                    
                    #if our estimate is incorrect, adjust the weights accordingly
                    if(trainingLabels[i] != guessedLabel):
                        calc = calcT(c, i)
                        
                        temp = trainingData[i]
                        tMaalF(calc, temp)                    
                    
                        cweights[c][trainingLabels[i]] = cweights[c][trainingLabels[i]] + temp
                        cweights[c][guessedLabel] = cweights[c][guessedLabel] - temp
        
        
        for c in Cgrid:
            for i in range(len(validationData)):
            #print "Starting validation ", i, "..."
            
                scores = util.Counter()
                
                for l in self.legalLabels:
                    #determine the score for a label for trainingData[i] and add it to a list of scores
                    score = util.Counter()
                    score = validationData[i] * cweights[c][l]
                    scores[l] = score
                
                #from the list of scores, get the label with the highest score
                guessedLabel = scores.argMax()
                
                
                #if our estimate is incorrect, adjust the weights accordingly
                if(validationLabels[i] == guessedLabel):
                    cscores[c] = cscores[c] + 1
        
        bestC = 0
        for c in Cgrid:
            if(cscores[c] > bestC):
                bestC = c
        
        self.weights = cweights[bestC]

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


