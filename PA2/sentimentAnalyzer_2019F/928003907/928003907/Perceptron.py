import sys
import getopt
import os
import math
import operator
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import random

class Perceptron:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
        """

        def __init__(self):
            self.train = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """

        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """Perceptron initialization"""
        self.numFolds = 10
        self.vectorizer = CountVectorizer()
        self.X = 0
        self.w = 0
        self.all_w = []
        self.final_w = 0
    #############################################################################
    # TODO TODO TODO TODO TODO
    # Implement the Perceptron classifier

    def classify(self, words):
        """ TODO
          'words' is a list of words to classify. Return 'pos' or 'neg' classification.
        """

        # Write code here
        test_arr = np.zeros(len(self.vectorizer.vocabulary_))
        for word in words:
            if word in self.vectorizer.vocabulary_:
                test_arr[self.vectorizer.vocabulary_[word]] += 1

        decide_sign = np.dot(self.final_w, test_arr)
        if decide_sign > 0:
            sign = 1
        elif decide_sign < 0:
            sign = -1
        else:
            sign = 0

        if sign>0:
            classType = 'pos'
        else:
            classType = 'neg'

        return classType
    
    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('pos' or 'neg') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier 
         * in the Perceptron class.
         * Returns nothing
        """

        # Write code here
        train_arr = np.zeros(len(self.vectorizer.vocabulary_))
        for word in words:
            if word in self.vectorizer.vocabulary_:
                train_arr[self.vectorizer.vocabulary_[word]] += 1

        if klass == 'pos':
            y = 1
        else:
            y = -1
        
        # if np.sin(np.dot(self.w, train_arr)) != y:
        decide_sign = np.dot(self.w, train_arr)
        if decide_sign > 0:
            sign = 1
        elif decide_sign < 0:
            sign = -1
        else:
            sign = 0

        # print(decide_sign,sign, y)
        if sign != y:
            self.w += (y-sign) * train_arr

    def train(self, split, iterations):
        """
        * TODO 
        * iterates through data examples
        * TODO 
        * use weight averages instead of final iteration weights
        """
        self.vectorizeWords(split)
        self.w = np.random.random(len(self.vectorizer.vocabulary_))
        for i in range(iterations):
            random.shuffle(split.train)
            for example in split.train:
                words = example.words
                self.addExample(example.klass, words)
            self.all_w.append(self.w)

        self.final_w = np.mean(self.all_w,axis=0)
    # END TODO (Modify code beyond here with caution)
    #############################################################################

    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here, 
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def vectorizeWords(self, split):
        train = []
        for sentence in split.train:
            sen = ''
            for word in sentence.words:
                sen += ' ' + word
            train.append(sen)

        self.X = self.vectorizer.fit_transform(train)

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile(
                    '%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile(
                    '%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            splits.append(split)
        return splits


def test10Fold(args):
    pt = Perceptron()

    iterations = int(args[1])
    splits = pt.crossValidationSplits(args[0])
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = Perceptron()
        accuracy = 0.0
        classifier.train(split, iterations)

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)


def classifyDir(trainDir, testDir, iter):
    classifier = Perceptron()
    trainSplit = classifier.trainSplit(trainDir)
    iterations = int(iter)
    classifier.train(trainSplit, iterations)
    testSplit = classifier.trainSplit(testDir)
    #testFile = classifier.readFile(testFilePath)
    accuracy = 0.0
    for example in testSplit.train:
        words = example.words
        guess = classifier.classify(words)
        if example.klass == guess:
            accuracy += 1.0
    accuracy = accuracy / len(testSplit.train)
    print('[INFO]\tAccuracy: %f' % accuracy)


def main():
    (options, args) = getopt.getopt(sys.argv[1:], '')

    if len(args) == 3:
        classifyDir(args[0], args[1], args[2])
    elif len(args) == 2:
        test10Fold(args)
    # test10Fold(['data/imdb1/','10'])

if __name__ == "__main__":
    main()
