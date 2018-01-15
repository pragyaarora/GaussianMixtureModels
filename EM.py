#!/usr/bin/python

#########################################################
# CSE 5523 starter code (HW#5)
# Alan Ritter
#########################################################

import random
import sys
import re
import numpy as np
import math
import  matplotlib.pyplot  as  plt
#GLOBALS/Constants
VAR_INIT = 1

def logExpSum(x):
    #TODO: implement logExpSum
    m = max(x)
    s = 0.0
    for p in x:
        s += np.exp(p - m)
    return m+math.log(s)

def readTrue(filename='wine-true.data'):
    f = open(filename)
    labels = []
    splitRe = re.compile(r"\s")
    for line in f:
        labels.append(int(splitRe.split(line)[0]))
    return labels

#########################################################################
#Reads and manages data in appropriate format
#########################################################################
class Data:
    def __init__(self, filename):
        self.data = []
        f = open(filename)
        (self.nRows,self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    #Computers the range of each column (returns a list of min-max tuples)
    def Range(self):
        ranges = []
        for j in range(self.nCols):
            min = self.data[0][j]
            max = self.data[0][j]
            for i in range(1,self.nRows):
                if self.data[i][j] > max:
                    max = self.data[i][j]
                if self.data[i][j] < min:
                    min = self.data[i][j]
            ranges.append((min,max))
        return ranges

    def __getitem__(self,row):
        return self.data[row]

#########################################################################
#Computes EM on a given data set, using the specified number of clusters
#self.parameters is a tuple containing the mean and variance for each gaussian
#########################################################################
class EM:
    def __init__(self, data, nClusters):
        #Initialize parameters randomly...
        random.seed()
        self.parameters = []
        self.priors = []        #Cluster priors
        self.nClusters = nClusters
        self.data = data
        ranges = data.Range()
        for i in range(nClusters):
            p = []
            initRow = random.randint(0,data.nRows-1)
            for j in range(data.nCols):
                #Randomly initalize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), VAR_INIT*(ranges[j][1] - ranges[j][0])))
            self.parameters.append(p)

        #Initialize priors uniformly
        for c in range(nClusters):
            self.priors.append(1/float(nClusters))

        self.parameterEstimation()


    def parameterEstimation(self):
        self.mean = []
        self.sdv = []
        for p in self.parameters:
            x = map(np.array, zip(*p))
            self.mean.append(x[0])
            self.sdv.append(np.diag(x[1]))
            #self.sdv.append(np.matrix(np.diag(x[1])))


    def LogLikelihood(self, data):
        logLikelihood = 0.0
        #TODO: compute log-likelihood of the data
        for i in range(data.nRows):
            p = []
            for j in range(self.nClusters):
                prob =  self.LogProb(i,j,data)
                p.append(prob)

            logLikelihood += logExpSum(p)

        return logLikelihood


    def CalProb(self,cluster,rowData):
        mat = np.matrix(self.sdv[cluster])
        det = np.linalg.det(mat)

        denominator = np.power(2*3.14, self.data.nRows)*det
        denominator = np.sqrt(denominator)

        sub = np.subtract(rowData, self.mean[cluster])
        inv = np.linalg.inv(mat)

        e = np.matmul(sub.T, inv)
        e = np.matmul(e, sub)
        e = np.exp(-0.5 * e)

        return  e/denominator


    #Compute marginal distributions of hidden variables
    def Estep(self):
        #TODO: E-step
        self.prob=[]
        for d in range(self.data.nRows):
            p = np.zeros(self.nClusters)
            for i in range(self.nClusters):
                p[i]=(self.LogProb(d,i,self.data))

            sum = logExpSum(p)
            p = [np.exp(x - sum) for x in p]
            self.prob.append(p)

    #Update the parameter estimates
    def Mstep(self):
        #TODO: M-step
        prob = np.array(self.prob)
        for i in range(self.nClusters):
            p = (prob[:,i]).sum()
            if p==0:
                p=0.000001
            self.priors[i] = p/(1.0 * self.data.nRows)

            sum = np.zeros(self.data.nCols)
            for d in range(self.data.nRows):
                sum += (np.multiply(prob[d][i], self.data[d]))
            self.mean[i] = sum / (p * 1.0)

            sum = np.zeros((self.data.nCols, self.data.nCols))
            for d in range(self.data.nRows):
                sub = np.subtract(self.data[d],self.mean[i])
                sum +=  (np.diag((prob[d][i] *sub) * sub))
            self.sdv[i] = sum / (p * 1.0)


    #Computes the probability that row was generated by cluster
    def LogProb(self, row, cluster, data):
        #TODO: compute probability row i was generated by cluster k

        const = -0.5 * (data.nCols * math.log(2 * np.pi))
        p = math.log(self.priors[cluster])

        sub = np.subtract(data[row], self.mean[cluster])
        diag = np.diagonal(self.sdv[cluster])
        det = 0
        length = len(diag)
        x = 0
        for i in range(length):
            value = diag[i]
            if value == 0:
                value = 0.000001
            det += math.log(value)
            x += math.pow(sub[i],2)/value

        det = -0.5 * det
        x = -0.5 * x
        ret = const+det+x+p
        return  ret

    def GetChangePercnt(self, old, new):
        value = (new-old)
        value /= old
        value *= 100
        return value

    def Run(self, maxsteps=100, testData=None):
        #TODO: Implement EM algorithm
        trainLikelihood = []
        testLikelihood = []
        iter = 0
        while iter < maxsteps:
            self.Estep()
            self.Mstep()
            logLikelihood = self.LogLikelihood(self.data)
            testLogLikelihood = self.LogLikelihood(testData)
            print ("Iteration : ",iter," Train Likelihood: ",logLikelihood," Test Likelihood: ",testLogLikelihood)
            trainLikelihood.append(logLikelihood)
            testLikelihood.append(testLogLikelihood)
            if iter != 0:
                if logLikelihood<trainLikelihood[iter-1]:
                    print("Bad Parameters")
                    raise Exception("Bad Parameters")
                if np.fabs(self.GetChangePercnt(trainLikelihood[iter-1], logLikelihood))<0.1:
                    break

            iter +=1
        return (trainLikelihood, testLikelihood)

    def Accuracy(self,trueLabels):
        data = self.prob
        length = len(data)
        labels = []
        count = 0
        dict = {}
        for i in range(length):
            value = np.argmax(data[i])
            labels.append(value)

            if dict.get(value) == None:
                dict[value] = []
            dict[value].append(i)

        for key,value in dict.iteritems():
            d = []
            for i in value:
                v = trueLabels[i]
                d.append(v)
            d = np.array(d)
            counts = np.bincount(d)
            c = np.argmax(counts)

            for i in value:
                labels[i] = c


        for i,j in zip(labels,trueLabels):
            if i==j:
                count+=1
        print count,"out of ",length
        accuracy = (count*1.0/length)*100
        return accuracy

    def GetTrueLabel(self,fileName):
        labels = []
        f = open(fileName)
        for line in f:
            x = line.split(" ")
            labels.append(int(x[0]))
        return labels


if __name__ == "__main__":
    d = Data('wine.train')

    train = []
    test = []
    acc = []
    #for i in range(1,11):
    # iter=0
    # for i in range(1,50):
    # while True:
    while True:
        try:
            if len(sys.argv) > 1:
                e = EM(d, int(sys.argv[1]))
            else:
                e = EM(d, 3)
            trainLikelihood,testLikelihood =  e.Run(100,Data('wine.test'))
        except Exception:
            continue
        break
    trueLabels = e.GetTrueLabel('wine-true.data')

    accuracy = e.Accuracy(trueLabels)
    print 'Accuracy :', accuracy

    #print iter/(1.0*50)

        # acc.append(accuracy)
        # train.append(trainLikelihood[len(trainLikelihood)-1])
        # test.append(testLikelihood[len(testLikelihood)-1])


    # plt.plot(trainLikelihood)
    # plt.ylabel('TrainingLikelihood')
    # plt.xlabel('Iterations')
    # plt.interactive(False)
    # plt.show()
    #
    # plt.plot(testLikelihood)
    # plt.ylabel('TestLikelihood')
    # plt.xlabel('Iterations')
    # plt.interactive(False)
    # plt.show()

    # plt.plot(acc)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Seeds')
    # plt.interactive(False)
    # plt.show()
    #
    # plt.plot(train)
    # plt.ylabel('TrainLikelihood')
    # plt.xlabel('Seeds')
    # plt.interactive(False)
    # plt.show()
    #
    # plt.plot(test)
    # plt.ylabel('TestLikelihood')
    # plt.xlabel('Seeds')
    # plt.interactive(False)
    # plt.show()

    # x = np.array([1,2,3,4,5,6,7,8,9,10])
    #
    # plt.plot(x,acc)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Clusters')
    # plt.interactive(False)
    # plt.show()
    #
    # plt.plot(x,train)
    # plt.ylabel('TrainingLikelihood')
    # plt.xlabel('Clusters')
    # plt.interactive(False)
    # plt.show()
    #
    # plt.plot(x,test)
    # plt.ylabel('TestLikelihood')
    # plt.xlabel('Clusters')
    # plt.interactive(False)
    # plt.show()



